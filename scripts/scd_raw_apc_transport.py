"""Copy packed BF16 KV between trusted benchmark workers on the same GPU."""

from __future__ import annotations

import math
import sys

import torch


def _packed_shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(size) is not int or size <= 0 for size in value) or value[-1] % 2):
        raise ValueError("expected a positive packed [layers, tokens, heads, 2 * head_dim] shape")
    return tuple(value)


def _device_uuid(index):
    identity = bytes(torch.cuda.get_device_properties(index).uuid.bytes)
    if len(identity) != 16 or not any(identity):
        raise RuntimeError("CUDA worker did not expose a physical GPU UUID")
    return identity


def _storage_args(values, shape, device_index):
    # PyTorch v2.12 and v2.14 StorageSharing.cpp use the same eight IPC fields.
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError("invalid CUDA IPC storage fields")
    index, memory_handle, size_bytes, offset_bytes, ref_handle, ref_offset, event_handle, sync = values
    if type(index) is not int or index != device_index:
        raise ValueError("CUDA IPC visible device differs from current worker device")
    if type(size_bytes) is not int or size_bytes != 2 * math.prod(shape):
        raise ValueError("packed BF16 shape differs from CUDA IPC storage byte count")
    if type(offset_bytes) is not int or not 0 <= offset_bytes <= sys.maxsize - size_bytes:
        raise ValueError("invalid CUDA IPC allocation offset")
    if type(memory_handle) is not bytes or not memory_handle:
        raise ValueError("missing CUDA IPC memory handle")
    if (type(ref_handle) is not bytes or not ref_handle or b"\x00" in ref_handle
            or type(ref_offset) is not int or not 0 <= ref_offset < 10000):
        raise ValueError("invalid CUDA IPC reference counter")
    if type(event_handle) is not bytes or len(event_handle) != 64 or type(sync) is not bool:
        raise ValueError("invalid CUDA IPC completion event")
    return tuple(values)


def export_payload(tensor: torch.Tensor) -> dict:
    """Export once per receiver and retain the source tensor through receive completion."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
        raise ValueError("expected a BF16 CUDA tensor")
    shape = _packed_shape(tensor.shape)
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        raise ValueError("expected contiguous packed KV with zero tensor storage offset")
    storage = tensor.untyped_storage()
    if storage.nbytes() != 2 * math.prod(shape):
        raise ValueError("packed KV must cover its entire storage")
    index = torch.cuda.current_device()
    if tensor.device.index != index:
        raise ValueError("export tensor is not on the current CUDA worker device")
    identity = _device_uuid(index)
    return {
        "shape": list(shape),
        "dtype": b"bfloat16",
        "device_uuid": identity,
        "storage": list(_storage_args(storage._share_cuda_(), shape, index)),
    }


def receive_payload(descriptor: dict, *, expected_shape=None) -> tuple[torch.Tensor, float]:
    """Consume an IPC descriptor and time a D2D copy into a private local tensor."""
    if type(descriptor) is not dict or descriptor.keys() != {"shape", "dtype", "device_uuid", "storage"}:
        raise ValueError("invalid packed KV descriptor fields")
    shape = _packed_shape(descriptor["shape"])
    dtype = descriptor["dtype"]
    if type(dtype) is not bytes or dtype != b"bfloat16":
        raise ValueError("expected a BF16 CUDA IPC descriptor")
    if expected_shape is not None and shape != _packed_shape(expected_shape):
        raise ValueError("packed CUDA IPC shape differs from the requested shape")
    index = torch.cuda.current_device()
    identity = descriptor["device_uuid"]
    if type(identity) is not bytes or identity != _device_uuid(index):
        raise ValueError("CUDA IPC sender and receiver are not on the same physical GPU")
    args = _storage_args(descriptor["storage"], shape, index)

    device = torch.device("cuda", index)
    with torch.cuda.device(device), torch.inference_mode(False), torch.no_grad():
        output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        storage = torch.UntypedStorage._new_shared_cuda(*args)
        if storage.device != device or storage.nbytes() != 2 * math.prod(shape):
            raise RuntimeError("imported storage device or byte count differs from the descriptor")
        # Identical to the storage binding used in torch._utils._rebuild_tensor.
        shared = torch.empty((0,), dtype=torch.bfloat16, device=device).set_(storage, 0, shape, output.stride())
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output.copy_(shared, non_blocking=True)
        stop.record()
        stop.synchronize()
        milliseconds = start.elapsed_time(stop)
        del shared, storage
    return output, milliseconds
