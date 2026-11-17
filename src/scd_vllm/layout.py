import torch
from scd_runtime.errors import UnsupportedProfile


def write_layer(dst, kv, slots, capability):
    k, v = kv
    n = slots.numel()
    h = k.shape[1]
    d = k.shape[-1]
    B = capability.block_size
    if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1 or k.shape[2] < n:
        raise UnsupportedProfile('invalid logical KV')
    if dst.ndim != 4 or tuple(dst.shape[1:]) != (h, B, 2 * d) or dst.dtype != torch.bfloat16:
        raise UnsupportedProfile('requires standardized [blocks,H,block_size,2D] BF16 cache view')
    if (slots.ndim != 1 or slots.dtype != torch.int64 or n == 0
            or bool((slots < 0).any()) or bool((slots >= dst.shape[0] * B).any())
            or slots.unique().numel() != n):
        raise UnsupportedProfile('invalid or repeated cache slots')
    indices = slots.to(dst.device)
    payload = torch.cat((k[0, :, :n].transpose(0, 1), v[0, :, :n].transpose(0, 1)), dim=-1).to(dst.device, dtype=dst.dtype)
    dst[indices // B, :, indices % B] = payload


def prepare_layer_writer(destinations, slots, capability):
    """Private synchronous resident writer; validate/upload one CPU slot map."""
    B = capability.block_size
    if (not isinstance(destinations, dict) or not destinations or type(B) is not int or B < 1
            or not isinstance(slots, torch.Tensor) or slots.device.type != 'cpu'
            or slots.ndim != 1 or slots.dtype != torch.int64 or slots.numel() == 0):
        raise UnsupportedProfile('requires private CPU int64 resident slots and destinations')
    first = next(iter(destinations.values()))
    if (not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[0] < 1
            or first.shape[1] < 1 or first.shape[2] != B or first.shape[3] < 2
            or first.shape[3] % 2 or first.dtype != torch.bfloat16):
        raise UnsupportedProfile('invalid resident destination geometry')

    def layout(t):
        return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

    if any(not isinstance(t, torch.Tensor) or t.shape != first.shape or t.dtype != first.dtype
           or t.device != first.device for t in destinations.values()):
        raise UnsupportedProfile('resident destinations must share geometry/device/dtype')
    approved = {id(t): layout(t) for t in destinations.values()}
    values = tuple(slots.tolist())
    n = len(values)
    if len(set(values)) != n or min(values) < 0 or max(values) >= first.shape[0] * B:
        raise UnsupportedProfile('invalid or repeated cache slots')
    try:
        version = slots._version
    except RuntimeError as exc:
        raise UnsupportedProfile('normal versioned CPU slots required') from exc
    with torch.inference_mode(False), torch.no_grad():
        indices = slots.to(first.device)
        blocks = indices // B
        offsets = indices % B

    def write(dst, kv, actual_slots, actual_capability):
        if (actual_slots is not slots or actual_capability is not capability or slots._version != version
                or not isinstance(dst, torch.Tensor) or approved.get(id(dst)) != layout(dst)):
            raise UnsupportedProfile('prepared resident slots or destination changed')
        k, v = kv
        h = dst.shape[1]
        d = dst.shape[-1] // 2
        if (k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1 or k.shape[1] != h
                or k.shape[2] < n or k.shape[3] != d or k.dtype != torch.bfloat16
                or v.dtype != k.dtype or k.device != dst.device or v.device != dst.device):
            raise UnsupportedProfile('invalid prepared resident logical KV')
        payload = torch.cat((k[0, :, :n].transpose(0, 1), v[0, :, :n].transpose(0, 1)), dim=-1)
        dst[blocks, :, offsets] = payload

    return write


def _extract_block_schedule(slots_seq: list[int] | tuple[int, ...], block_size: int, max_blocks: int):
    """Detect standard vLLM block-sequential slot maps for page-level index_copy_.

    Returns (is_block_sequential, full_block_ids, tail_block_id, rem).
    Validates bounds and uniqueness on CPU in O(n) without GPU syncs.
    """
    n = len(slots_seq)
    if n == 0:
        raise UnsupportedProfile('empty slot sequence')
    max_slot = max_blocks * block_size
    if len(set(slots_seq)) != n or min(slots_seq) < 0 or max(slots_seq) >= max_slot:
        raise UnsupportedProfile('invalid or repeated cache slots')

    full_blocks = n // block_size
    rem = n % block_size
    full_block_ids = []
    for b_idx in range(full_blocks):
        base_pos = b_idx * block_size
        first_slot = slots_seq[base_pos]
        if first_slot % block_size != 0:
            return False, (), None, 0
        blk_id = first_slot // block_size
        for off in range(1, block_size):
            if slots_seq[base_pos + off] != first_slot + off:
                return False, (), None, 0
        full_block_ids.append(blk_id)

    tail_blk_id = None
    if rem > 0:
        base_pos = full_blocks * block_size
        first_slot = slots_seq[base_pos]
        if first_slot % block_size != 0:
            return False, (), None, 0
        tail_blk_id = first_slot // block_size
        for off in range(1, rem):
            if slots_seq[base_pos + off] != first_slot + off:
                return False, (), None, 0

    return True, tuple(full_block_ids), tail_blk_id, rem


def write_packed_layers_fast(destinations: dict[int, torch.Tensor], packed_kv: torch.Tensor, slots_seq, capability) -> None:
    """Scatter [L, N_cap, H, 2*D] BF16 packed KV directly into vLLM paged cache.

    Uses page-aligned index_copy_ for complete blocks and direct slice copy for
    a partial tail block (zero per-layer torch.cat, zero 2D advanced indexing
    when block-sequential, and zero host-blocking synchronize).
    """
    B = capability.block_size
    if not isinstance(destinations, dict) or not destinations or type(B) is not int or B < 1:
        raise UnsupportedProfile('invalid destinations or block_size')
    L = len(destinations)
    first = destinations[0]
    if (not isinstance(first, torch.Tensor) or first.ndim != 4 or first.dtype != torch.bfloat16
            or first.shape[2] != B):
        raise UnsupportedProfile('invalid destination KV cache tensor')
    H = first.shape[1]
    two_d = first.shape[3]
    slots_list = list(slots_seq)
    n = len(slots_list)
    if (not isinstance(packed_kv, torch.Tensor) or packed_kv.ndim != 4
            or packed_kv.shape[0] != L or packed_kv.shape[1] < n
            or packed_kv.shape[2] != H or packed_kv.shape[3] != two_d
            or packed_kv.dtype != torch.bfloat16 or packed_kv.device != first.device):
        raise UnsupportedProfile('packed KV geometry/dtype/device differs from destination cache')

    is_seq, full_blk_ids, tail_blk_id, rem = _extract_block_schedule(slots_list, B, first.shape[0])
    dev = first.device

    with torch.no_grad():
        if is_seq:
            full_tokens = len(full_blk_ids) * B
            if full_blk_ids:
                blk_idx_t = torch.tensor(full_blk_ids, dtype=torch.int64, device=dev)
                # Single batched transpose across all L layers: [L, full_blocks, B, H, 2D] -> [L, full_blocks, H, B, 2D]
                paged_src = packed_kv[:, :full_tokens, :, :].reshape(L, len(full_blk_ids), B, H, two_d).transpose(2, 3).contiguous()
                for layer in range(L):
                    destinations[layer].index_copy_(0, blk_idx_t, paged_src[layer])
            if rem > 0 and tail_blk_id is not None:
                # Single batched tail transpose: [L, rem, H, 2D] -> [L, H, rem, 2D]
                tail_src = packed_kv[:, full_tokens:n, :, :].transpose(1, 2)
                for layer in range(L):
                    destinations[layer][tail_blk_id, :, :rem, :].copy_(tail_src[layer])
        else:
            indices = torch.tensor(slots_list, dtype=torch.int64, device=dev)
            blocks = indices // B
            offsets = indices % B
            for layer in range(L):
                destinations[layer][blocks, :, offsets] = packed_kv[layer, :n]
