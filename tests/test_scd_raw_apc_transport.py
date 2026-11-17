from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.fixture
def ipc_runtime(monkeypatch):
    from scripts import scd_raw_apc_transport as transport

    trace = []
    shape = (2, 3, 2, 4)
    bfloat16 = object()
    device = SimpleNamespace(type="cuda", index=0)
    gpu_uuid = b"physical-gpu-id!"
    handle = (0, b"A" * 64, 96, 512, b"/torch_test_ref", 3, b"E" * 64, True)

    class Storage:
        def __init__(self, data):
            self.data = data
            self.device = device

        def nbytes(self):
            return 96

        def _share_cuda_(self):
            trace.append("export")
            return handle

        @classmethod
        def _new_shared_cuda(cls, *values):
            assert values == handle
            trace.append("import")
            return shared_storage

    class Tensor:
        def __init__(self, dims, storage=None):
            self.dtype = bfloat16
            self.device = device
            self.shape = dims
            self.storage = storage
            self.offset = 0

        def storage_offset(self):
            return self.offset

        def untyped_storage(self):
            return self.storage

        def is_contiguous(self):
            return True

        def stride(self):
            return (24, 8, 4, 1)

        def set_(self, storage, offset, dims, stride):
            assert dims == shape and stride == (24, 8, 4, 1) and offset == 0
            self.storage = storage
            self.shape = dims
            return self

        def copy_(self, source, *, non_blocking):
            assert source.storage is shared_storage and non_blocking
            self.storage.data[:] = source.storage.data
            trace.append("copy")

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            trace.append("record")

        def synchronize(self):
            trace.append("synchronize")

        def elapsed_time(self, stop):
            assert trace[-1] == "synchronize"
            return 0.125

    shared_storage = Storage(bytearray(range(96)))
    fake_torch = SimpleNamespace(
        Tensor=Tensor,
        bfloat16=bfloat16,
        UntypedStorage=Storage,
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda idx: SimpleNamespace(uuid=SimpleNamespace(bytes=gpu_uuid)),
            device=lambda dev: nullcontext(),
            Event=Event,
        ),
        device=lambda kind, idx: device,
        empty=lambda dims, **options: Tensor(dims, Storage(bytearray(96))),
        inference_mode=lambda enabled: nullcontext(),
        no_grad=nullcontext,
    )
    monkeypatch.setattr(transport, "torch", fake_torch)
    return transport, Tensor(shape, shared_storage), trace


def test_ipc_receive_copies_bf16_bytes_before_returning(ipc_runtime):
    transport, source, trace = ipc_runtime
    descriptor = transport.export_payload(source)
    assert descriptor["shape"] == [2, 3, 2, 4]
    assert descriptor["dtype"] == b"bfloat16"
    assert isinstance(descriptor["storage"], list)

    output, milliseconds = transport.receive_payload(descriptor, expected_shape=source.shape)

    assert trace == ["export", "import", "record", "copy", "record", "synchronize"]
    assert milliseconds == 0.125
    assert output.storage is not source.storage
    assert output.storage.data == source.storage.data


@pytest.mark.parametrize("invalid", ["dtype", "shape", "device_uuid"])
def test_ipc_rejects_mismatched_descriptors_before_opening(ipc_runtime, invalid):
    transport, source, trace = ipc_runtime
    descriptor = transport.export_payload(source)
    descriptor[invalid] = {"dtype": b"float16", "shape": [2, 1, 2, 4], "device_uuid": b"wrong-gpu-uuid!!"}[invalid]
    with pytest.raises(ValueError):
        transport.receive_payload(descriptor, expected_shape=source.shape)
    assert trace == ["export"]


def test_ipc_export_rejects_nonzero_tensor_storage_offset(ipc_runtime):
    transport, source, trace = ipc_runtime
    source.offset = 1
    with pytest.raises(ValueError, match="storage offset"):
        transport.export_payload(source)
    assert trace == []
