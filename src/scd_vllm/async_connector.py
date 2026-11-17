"""Asynchronous event transfer integration shared by file and resident connectors."""
import re
import torch
from .async_transfer import AsyncKVTransfer, CudaTransferBackend, prepare_layer_copy
from scd_runtime.errors import UnsupportedProfile


class EventTransferAdapter:
    def __init__(self, capability):
        self.capability = capability
        self.owner = None
        self._received = set()
        self._failed = set()
        self._invalid = set()
        self.cancelled = set()

    def _get_owner(self, destinations):
        device = next(iter(destinations.values())).device
        if device.type != 'cuda':
            raise UnsupportedProfile('real asynchronous connector requires CUDA destinations')
        if self.owner is None:
            self.owner = AsyncKVTransfer(CudaTransferBackend(device))
        elif self.owner.backend.device != device:
            raise UnsupportedProfile('transfer owner cannot change CUDA device')
        return self.owner

    def fail_without_writes(self, request_id, slots, release):
        release()
        self._received.add(request_id)
        self._failed.add(request_id)
        self._invalid.update(value // self.capability.block_size for value in slots)

    def submit(self, request_id, lease_id, kv, destinations, slots, *, on_complete, release):
        owner = self._get_owner(destinations)
        indices = torch.tensor(slots, dtype=torch.int64, device='cpu')
        writes = [prepare_layer_copy(destinations[layer], pair, indices, self.capability)
                  for layer, pair in enumerate(kv.layers)]
        ready = (owner.backend.record_source_ready(),) if kv.layers[0][0].is_cuda else ()
        ticket = owner.begin(
            request_id, lease_id,
            [value // self.capability.block_size for value in slots],
            [str(layer) for layer in destinations],
            retain=(kv, destinations, indices, *writes),
            release=release, on_complete=on_complete,
            asynchronous=True, source_ready=ready,
        )
        try:
            for layer, write in enumerate(writes):
                owner.submit_layer(ticket, str(layer), write)
            owner.seal(ticket)
        except BaseException:
            owner.cancel(request_id)
            raise
        return ticket

    def cancel(self, request_ids):
        self.cancelled.update(request_ids)
        if self.owner is not None:
            for rid in request_ids:
                self.owner.cancel(rid)

    def collect(self, finished_req_ids):
        if self.owner is not None:
            result = self.owner.poll(finished_req_ids)
            self._received.update(result.received)
            self._failed.update(result.failed)
            self._invalid.update(result.invalid_blocks)
        received, failed = self._received, self._failed
        self._received, self._failed = set(), set()
        self.cancelled.difference_update(received)
        return received, failed

    def pending(self, request_id):
        return self.owner is not None and request_id in self.owner._jobs

    def lease_owned_by_other(self, request_id, lease_id):
        if self.owner is None:
            return False
        with self.owner._lock:
            ticket = self.owner._leases.get(lease_id)
            return ticket is not None and ticket.request_id != request_id

    def errors(self):
        invalid, self._invalid = self._invalid, set()
        return invalid

    def wait_layer(self, layer_name):
        if self.owner is not None:
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', layer_name)
            if match is not None:
                self.owner.wait_for_layer(match.group(1))

    def shutdown(self):
        if self.owner is not None:
            self.owner.shutdown()


def initialize_request_mode(connector, extra, profile):
    asynchronous = extra.get_from_extra_config('async_loading', False)
    if type(asynchronous) is not bool:
        raise UnsupportedProfile('async_loading requires an explicit boolean')
    connector._async_loading = asynchronous
    connector._events = EventTransferAdapter(connector._cache_contract()[1]) if asynchronous else None
    connector._cancelled_requests = set()


def add_routing_metadata(connector, meta, scheduler_output):
    meta.cancelled = set(connector._cancelled_requests)
    connector._cancelled_requests.clear()


def bind_routing_metadata(connector, meta, profile):
    if connector._events is not None:
        pending = {rid for rid, *_ in meta.loads}
        connector._events.cancel({
            rid for rid in meta.cancelled
            if rid in pending or connector._events.pending(rid)
        })


class EventConnectorMixin:
    def bind_connector_metadata(self, connector_metadata):
        super().bind_connector_metadata(connector_metadata)
        profile, _ = self._cache_contract()
        bind_routing_metadata(self, connector_metadata, profile)

    def get_transfer_results(self, finished_req_ids):
        if self._events is None:
            return super().get_transfer_results(finished_req_ids)
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
        received, failed = self._events.collect(finished_req_ids)
        return KVConnectorTransferResults(finished_recving=received, failed_recving=failed)

    def shutdown(self):
        if self._events is not None:
            self.owner_shutdown()

    def owner_shutdown(self):
        if self._events is not None:
            self._events.shutdown()
