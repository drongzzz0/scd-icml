from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency in release checks
    torch = None

from scd_repro.adapters import TransformersCaptureSession


@unittest.skipIf(torch is None, "torch is not installed")
class TransformersCaptureStoreDeviceTest(unittest.TestCase):
    def _session(self, capture_store_device: str, target_device: str = "cpu") -> TransformersCaptureSession:
        session = TransformersCaptureSession.__new__(TransformersCaptureSession)
        session.torch = torch
        session.capture_store_device = capture_store_device
        session.target_device = target_device
        return session

    def test_cpu_capture_store_materializes_float32_contiguous_cpu_tensor(self) -> None:
        session = self._session("cpu")
        tensor = torch.ones((2, 3), dtype=torch.float16).t()

        out = session._materialize_capture_tensor(tensor)

        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(out.is_contiguous())

    def test_model_capture_store_uses_target_device(self) -> None:
        session = self._session("model", target_device="cpu")
        tensor = torch.ones((2, 3), dtype=torch.float16)

        out = session._materialize_capture_tensor(tensor)

        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(out.is_contiguous())


if __name__ == "__main__":
    unittest.main()
