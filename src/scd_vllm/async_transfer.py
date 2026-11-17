"""Worker-owned KV copies with event-based completion and lease retention.

This module does not schedule requests. For between-step receives the connector
returns ``is_async=True`` to the existing scheduler, which retains destination
blocks until ``received`` is reported. A cancelled receive still completes on
that receive channel, never on the send channel. Synchronous imports may use
the same copy stream with per-layer GPU dependencies and no receive message.

Only this owner may submit writes. Keep all preparation callbacks behind its
generation ticket; a callback that writes directly to the cache is unsupported.
Producers using another CUDA stream must pass their recorded source-ready event
to begin(). Tensor record_stream() is allocation bookkeeping, not that data
dependency. Registered destinations must not be replaced while transfers live.
"""
from dataclasses import dataclass, field
import threading
from typing import Any, Callable


class TransferError(RuntimeError):
    """A receive failed; its outstanding writes still need to drain."""


class TransferStateError(TransferError):
    """An invalid, cancelled, or stale submission was rejected before writing."""


class UnsafeTransferError(TransferError):
    """Completion cannot be established; retain resources and fail closed."""


@dataclass(frozen=True)
class TransferTicket:
    request_id: str
    lease_id: str
    generation: int


@dataclass
class TransferCompletions:
    received: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    invalid_blocks: set[int] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class _Transfer:
    ticket: TransferTicket
    blocks: frozenset[int]
    expected_layers: frozenset[str]
    release: Callable[[], None]
    on_complete: Callable[[], None] | None
    asynchronous: bool
    retained: list[Any]
    layers: dict[str, Any] = field(default_factory=dict)
    phase: str = 'OPEN'
    final_event: Any = None
    error: str | None = None
    release_error: str | None = None


class CudaTransferBackend:
    """One private stream. Polling never synchronizes the host or whole device."""

    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA transfer backend requires a CUDA device')
        self.stream = torch.cuda.Stream(device=self.device)

    def order_after_current(self, source_ready=()):
        # Also orders after runner zeroing/writes already submitted on this
        # step's current compute stream. Extra producer streams need explicit
        # events: waiting on the current stream alone cannot cover them.
        self.stream.wait_stream(self.torch.cuda.current_stream(self.device))
        for event in source_ready:
            self.stream.wait_event(event)

    def record_source_ready(self):
        """Call on the producer's stream, after its final source write."""
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        return event

    def enqueue(self, writer):
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.stream):
            writer()
            return self.fence()

    def fence(self):
        event = self.torch.cuda.Event()
        event.record(self.stream)
        return event

    def wait(self, event):
        self.torch.cuda.current_stream(self.device).wait_event(event)

    def drain(self):
        # Only shutdown/error recovery may block the host, and only for this
        # owner's stream. No torch.cuda.synchronize(device) here.
        self.stream.synchronize()


class AsyncKVTransfer:
    """Serialize host submission and cancellation; retain every in-flight lease.

    begin/submit_layer/seal are host submission only. poll() queries events and
    emits one atomic failure/completion snapshot. A failed event query or lease
    release poisons the owner: it never advertises safe reuse without evidence.
    on_complete runs only after a successful fence, before release; use it to
    record a real injection receipt. Cancellation never records such a receipt.
    """

    def __init__(self, backend):
        self.backend = backend
        self._lock = threading.RLock()
        self._jobs: dict[str, _Transfer] = {}
        self._blocks: dict[int, TransferTicket] = {}
        self._leases: dict[str, TransferTicket] = {}
        self._generation = 0
        self._closed = False
        self._poisoned = False
        self._polling = False
        self._submitting = False

    def _guard_submission(self):
        # RLock serializes other threads but permits same-thread callbacks.
        # A writer has not yet established its final fence: retiring or
        # cancelling it here could free blocks before it queues its last write.
        if self._submitting:
            raise TransferStateError('lifecycle reentry during layer submission is forbidden')

    def _available(self):
        self._guard_submission()
        if self._poisoned:
            raise UnsafeTransferError('transfer owner is poisoned; resources retained')
        if self._closed:
            raise TransferStateError('transfer owner is closed')

    def _job(self, ticket):
        job = self._jobs.get(ticket.request_id)
        if job is None or job.ticket != ticket:
            raise TransferStateError('stale transfer generation or lease')
        return job

    def begin(self, request_id, lease_id, block_ids, layer_names, *, retain=(),
              release, on_complete=None, asynchronous=True, source_ready=()):
        with self._lock:
            self._available()
            if not isinstance(request_id, str) or not request_id:
                raise ValueError('nonempty request ID required')
            if not isinstance(lease_id, str) or not lease_id:
                raise ValueError('nonempty lease ID required')
            if request_id in self._jobs:
                raise TransferStateError('request already owns a transfer')
            if lease_id in self._leases:
                raise TransferStateError('lease already belongs to an in-flight transfer')
            blocks = frozenset(block_ids)
            names = tuple(layer_names)
            if not blocks or any(type(x) is not int or x < 0 for x in blocks):
                raise ValueError('nonempty nonnegative block IDs required')
            if (not names or len(set(names)) != len(names)
                    or any(not isinstance(n, str) or not n for n in names)):
                raise ValueError('unique nonempty layer names required')
            if type(asynchronous) is not bool or not callable(release):
                raise ValueError('explicit asynchronous boolean and release callback required')
            if on_complete is not None and not callable(on_complete):
                raise ValueError('completion callback must be callable')
            if any(block in self._blocks for block in blocks):
                raise TransferStateError('destination block still owned by an in-flight transfer')
            self._generation += 1
            ticket = TransferTicket(request_id, lease_id, self._generation)
            ready = tuple(source_ready)
            job = _Transfer(ticket, blocks, frozenset(names), release,
                            on_complete, asynchronous, list(retain) + list(ready))
            self._jobs[request_id] = job
            self._blocks.update((block, ticket) for block in blocks)
            self._leases[lease_id] = ticket
            try:
                self.backend.order_after_current(ready)
            except Exception as exc:
                self._fail(job, exc)
                raise TransferError(str(exc)) from exc
            return ticket

    def hold(self, ticket, *objects):
        with self._lock:
            self._available()
            job = self._job(ticket)
            if job.phase != 'OPEN':
                raise TransferStateError('cannot retain for a cancelled/sealed transfer')
            job.retained.extend(objects)

    def _fence(self, job):
        try:
            job.final_event = self.backend.fence()
        except Exception as exc:
            self._poisoned = True
            raise UnsafeTransferError('cannot fence outstanding writes; resources retained') from exc

    def _fail(self, job, exc):
        job.phase = 'FAILED'
        job.error = f'{type(exc).__name__}: {exc}'
        self._fence(job)

    def submit_layer(self, ticket, layer_name, writer):
        with self._lock:
            self._available()
            job = self._job(ticket)
            if job.phase != 'OPEN':
                raise TransferStateError('cancelled or sealed transfer cannot submit writes')
            if layer_name not in job.expected_layers or layer_name in job.layers:
                raise TransferStateError('unexpected or duplicate layer submission')
            if not callable(writer):
                raise ValueError('layer writer must be callable')
            # Includes destination, source/staging and immutable slots captured
            # by the prepared writer, even after host submission returns.
            job.retained.append(writer)
            self._submitting = True
            try:
                job.layers[layer_name] = self.backend.enqueue(writer)
            except Exception as exc:
                self._submitting = False
                # The writer might have queued a partial copy before raising.
                # Fence that work before reporting completion or freeing slots.
                self._fail(job, exc)
                raise TransferError(str(exc)) from exc
            finally:
                self._submitting = False

    def seal(self, ticket):
        with self._lock:
            self._available()
            job = self._job(ticket)
            if job.phase != 'OPEN':
                raise TransferStateError('cancelled or sealed transfer cannot be sealed')
            if set(job.layers) != job.expected_layers:
                exc = TransferStateError('missing layers in KV transfer')
                self._fail(job, exc)
                raise exc
            self._fence(job)
            job.phase = 'SEALED'

    def cancel(self, request_id):
        with self._lock:
            self._guard_submission()
            job = self._jobs.get(request_id)
            if job is None or job.phase in ('CANCELLED', 'FAILED', 'RETIRING', 'RELEASE_FAILED'):
                return
            job.phase = 'CANCELLED'
            job.error = 'receive cancelled'
            if job.final_event is None:
                self._fence(job)

    def wait_for_layer(self, layer_name, *, tickets=None):
        with self._lock:
            self._available()
            # Async-between-step receives are not inputs to an unrelated
            # current forward. Scheduler admission waits for full completion.
            jobs = ([self._job(ticket) for ticket in tickets] if tickets is not None
                    else [j for j in self._jobs.values() if not j.asynchronous])
            for job in jobs:
                if job.phase in ('FAILED', 'CANCELLED'):
                    raise TransferError(job.error)
                if layer_name not in job.layers:
                    raise TransferStateError('required layer has not been submitted')
                self.backend.wait(job.layers[layer_name])

    def _retire(self, job, result, *, publish=True):
        failed = job.phase in ('FAILED', 'CANCELLED')
        job.phase = 'RETIRING'
        if not failed and publish and job.on_complete is not None:
            try:
                job.on_complete()
            except Exception as exc:
                failed = True
                job.error = f'completion callback failed: {type(exc).__name__}: {exc}'
        if job.release_error is not None:
            raise UnsafeTransferError(job.release_error)
        try:
            job.release()
        except Exception as exc:
            job.release_error = f'lease release failed: {type(exc).__name__}: {exc}'
            job.phase = 'RELEASE_FAILED'
            self._poisoned = True
            raise UnsafeTransferError(job.release_error) from exc
        if publish:
            if job.asynchronous:
                result.received.add(job.ticket.request_id)
                if failed:
                    result.failed.add(job.ticket.request_id)
            if failed:
                result.invalid_blocks.update(job.blocks)
                result.errors[job.ticket.request_id] = job.error or 'transfer failed'
        del self._jobs[job.ticket.request_id]
        del self._leases[job.ticket.lease_id]
        for block in job.blocks:
            if self._blocks.get(block) == job.ticket:
                del self._blocks[block]
        job.retained.clear()

    def poll(self, finished_req_ids=()):
        with self._lock:
            self._available()
            for rid in finished_req_ids:
                self.cancel(rid)
            result = TransferCompletions()
            # Completion callbacks may inspect status, but must not recursively
            # consume another completion snapshot or release a lease twice.
            if self._polling:
                return result
            self._polling = True
            try:
                for job in list(self._jobs.values()):
                    if job.final_event is None or job.phase == 'RETIRING':
                        continue
                    try:
                        done = job.final_event.query()
                    except Exception as exc:
                        self._poisoned = True
                        raise UnsafeTransferError('event query failed; resources retained') from exc
                    if done:
                        self._retire(job, result)
                return result
            finally:
                self._polling = False

    def shutdown(self):
        """Drain this owner's events, then release; never advertise new success.

        A fatal CUDA error or failed release leaves references and reservations
        retained and raises. There is no optimistic cleanup or automatic retry
        of a release callback with unknown external side effects.
        """
        with self._lock:
            self._guard_submission()
            self._closed = True
            for rid in list(self._jobs):
                self.cancel(rid)
            for job in list(self._jobs.values()):
                if job.phase == 'RETIRING':
                    continue
                try:
                    if job.final_event is None:
                        self.backend.drain()
                    else:
                        job.final_event.synchronize()
                except Exception as exc:
                    self._poisoned = True
                    raise UnsafeTransferError('shutdown could not drain writes; resources retained') from exc
                self._retire(job, TransferCompletions(), publish=False)


def prepare_layer_copy(dst, kv, slots, capability):
    """Validate on CPU, own slots/staging, return a stream-submitted layer write.

    CPU slots are required so validation never reads CUDA values on the host.
    CPU source data is validated and packed into private pinned staging before
    submission. CUDA source geometry/version is checked without readback, and
    nonfinite values raise via a device assertion on the copy stream. Sources
    must remain immutable; their producer events belong in begin(source_ready).
    CPU destinations are supported only as an exact-layout unit-test reference.
    """
    import torch
    if not isinstance(dst, torch.Tensor) or not isinstance(slots, torch.Tensor):
        raise ValueError('tensor destination and CPU slots required')
    if not isinstance(kv, (tuple, list)) or len(kv) != 2:
        raise ValueError('logical K/V pair required')
    k, v = kv
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError('logical K/V tensors required')
    B = capability.block_size
    if type(B) is not int or B <= 0:
        raise ValueError('positive block size required')
    if (slots.device.type != 'cpu' or slots.dtype != torch.int64
            or slots.ndim != 1 or slots.numel() == 0):
        raise ValueError('nonempty one-dimensional CPU int64 slots required')
    owned_slots = slots.detach().clone().contiguous()
    values = owned_slots.tolist()  # CPU only: never an implicit GPU readback.
    n = len(values)
    if (dst.ndim != 4 or dst.dtype != torch.bfloat16
            or dst.device.type not in ('cpu', 'cuda') or dst.shape[0] <= 0
            or k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
            or k.shape[1] <= 0 or k.shape[-1] <= 0 or k.shape[2] < n
            or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16
            or k.device != v.device or k.device.type not in ('cpu', 'cuda')
            or tuple(dst.shape[1:]) != (k.shape[1], B, 2 * k.shape[-1])):
        raise ValueError('logical/registered KV geometry, dtype or device differs')
    if len(set(values)) != n or min(values) < 0 or max(values) >= dst.shape[0] * B:
        raise ValueError('invalid or repeated cache slots')
    if k.device.type == 'cuda' and k.device != dst.device:
        raise ValueError('CUDA source and destination must share a device')
    def pack():
        return torch.cat((k[0, :, :n].transpose(0, 1),
                          v[0, :, :n].transpose(0, 1)), dim=-1).contiguous()
    staging = None
    if k.device.type == 'cpu':
        if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):
            raise ValueError('nonfinite logical KV')
        staging = pack()
        if dst.device.type == 'cuda':
            staging = staging.pin_memory()
    elif not hasattr(torch, '_assert_async'):
        raise ValueError('nonblocking CUDA finite assertion unavailable')
    if dst.device.type == 'cuda':
        owned_slots = owned_slots.pin_memory()
    versions = (k._version, v._version) if k.device.type == 'cuda' else None

    def write():
        with torch.no_grad():
            if versions is not None:
                if (k._version, v._version) != versions:
                    raise TransferStateError('CUDA source mutated after preparation')
                torch._assert_async(torch.isfinite(k).all(), 'nonfinite logical K')
                torch._assert_async(torch.isfinite(v).all(), 'nonfinite logical V')
            payload = staging if staging is not None else pack()
            indices = owned_slots.to(dst.device, non_blocking=dst.device.type == 'cuda')
            payload = payload.to(dst.device, non_blocking=dst.device.type == 'cuda')
            dst[indices // B, :, indices % B] = payload
    return write
