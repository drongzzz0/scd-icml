from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scd_repro.cli.capture_pair as capture_pair_module  # noqa: E402
from scd_repro.adapters import TransformersCaptureSession  # noqa: E402
from scd_repro.torch_utils import require_torch  # noqa: E402


class CapturePairSequentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = require_torch()

    def test_should_capture_sequentially_only_when_runtime_devices_match(self) -> None:
        settings = {
            "model_a_device": "cuda:0",
            "model_b_device": "cuda:0",
            "cpu_fallback": True,
        }
        with mock.patch.object(capture_pair_module, "resolve_device", side_effect=lambda device, cpu_fallback=True: f"norm::{device}"):
            self.assertTrue(capture_pair_module._should_capture_sequentially(settings))
            settings["model_b_device"] = "cuda:1"
            self.assertFalse(capture_pair_module._should_capture_sequentially(settings))

    def test_capture_payload_roundtrip_preserves_prompts_and_tensors(self) -> None:
        payload = {
            0: {
                "k": self.torch.arange(4, dtype=self.torch.float32).reshape(1, 1, 2, 2),
                "v": self.torch.ones((1, 1, 2, 2), dtype=self.torch.float32),
            }
        }
        prompts = ["hello", "world"]
        tmp_root = REPO_ROOT / ".tmp_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        path = tmp_root / f"capture_pair_roundtrip_{uuid.uuid4().hex}.pt"
        try:
            capture_pair_module._save_capture_payload(path, prompts, payload, "bfloat16")
            loaded_prompts, loaded_payload = capture_pair_module._load_capture_payload(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(loaded_prompts, prompts)
        self.assertEqual(sorted(loaded_payload.keys()), [0])
        self.assertEqual(tuple(loaded_payload[0]["k"].shape), (1, 1, 2, 2))
        self.assertEqual(loaded_payload[0]["k"].dtype, self.torch.bfloat16)

    def test_close_drops_model_and_layer_references(self) -> None:
        removed = []

        class DummyHook:
            def remove(self) -> None:
                removed.append(True)

        session = TransformersCaptureSession.__new__(TransformersCaptureSession)
        session.hooks = [DummyHook(), DummyHook()]
        session.xnorm_by_layer = {0: self.torch.ones((1,), dtype=self.torch.float32)}
        session.residual_by_layer = {0: self.torch.ones((1,), dtype=self.torch.float32)}
        session.layers = [object()]
        session.tokenizer = object()
        session.model = object()

        session.close()

        self.assertEqual(len(removed), 2)
        self.assertEqual(session.hooks, [])
        self.assertEqual(session.xnorm_by_layer, {})
        self.assertEqual(session.residual_by_layer, {})
        self.assertEqual(session.layers, [])
        self.assertIsNone(session.tokenizer)
        self.assertIsNone(session.model)


if __name__ == "__main__":
    unittest.main()
