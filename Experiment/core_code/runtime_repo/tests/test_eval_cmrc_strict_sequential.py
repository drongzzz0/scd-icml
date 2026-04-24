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

import scd_repro.cli.eval_cmrc_strict as eval_cmrc_strict_module  # noqa: E402
from scd_repro.strict_runtime import PrefillCapture  # noqa: E402
from scd_repro.strict_runtime import TransformersRuntimeSession  # noqa: E402
from scd_repro.torch_utils import require_torch  # noqa: E402


class EvalCMRCStrictSequentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = require_torch()

    def test_should_use_sequential_runtime_when_devices_normalize_equal(self) -> None:
        with mock.patch.object(
            eval_cmrc_strict_module,
            "resolve_device",
            side_effect=lambda device, cpu_fallback=True: f"norm::{device}",
        ):
            self.assertTrue(eval_cmrc_strict_module._should_use_sequential_runtime("cuda:0", "cuda:0"))
            self.assertFalse(eval_cmrc_strict_module._should_use_sequential_runtime("cuda:0", "cuda:1"))
            self.assertTrue(eval_cmrc_strict_module._should_use_sequential_runtime("cuda:0", "cuda:1", force=True))

    def test_prefill_capture_roundtrip_moves_tensors_to_cpu(self) -> None:
        capture = PrefillCapture(
            input_ids=self.torch.tensor([[1, 2, 3]], dtype=self.torch.long),
            attention_mask=self.torch.ones((1, 3), dtype=self.torch.long),
            logits_last=self.torch.randn((1, 32000), dtype=self.torch.float32),
            past_key_values=[
                (
                    self.torch.randn((1, 1, 3, 2), dtype=self.torch.float32),
                    self.torch.randn((1, 1, 3, 2), dtype=self.torch.float32),
                )
            ],
            xnorm_by_layer={0: self.torch.randn((1, 3, 4), dtype=self.torch.float32)},
            prefill_ms=12.5,
            layer_ms=[1.0],
            vocab_size=32000,
            residual_by_layer={0: self.torch.randn((1, 3, 4), dtype=self.torch.float32)},
        )
        tmp_root = REPO_ROOT / ".tmp_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        path = tmp_root / f"eval_cmrc_strict_prefill_{uuid.uuid4().hex}.pt"
        try:
            eval_cmrc_strict_module._save_prefill_capture(path, capture)
            loaded = eval_cmrc_strict_module._load_prefill_capture(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(tuple(loaded.input_ids.shape), (1, 3))
        self.assertEqual(str(loaded.input_ids.device), "cpu")
        self.assertEqual(str(loaded.past_key_values[0][0].device), "cpu")
        self.assertEqual(sorted(loaded.xnorm_by_layer.keys()), [0])
        self.assertEqual(sorted(loaded.residual_by_layer.keys()), [0])
        self.assertEqual(loaded.prefill_ms, 12.5)
        self.assertEqual(loaded.layer_ms, [1.0])
        self.assertEqual(loaded.vocab_size, 32000)

    def test_method_dependencies_skip_patch_for_baseline_and_reuse_only(self) -> None:
        deps = eval_cmrc_strict_module._method_dependencies(["full_consumer", "reuse_only"])
        self.assertTrue(deps["needs_reuse"])
        self.assertFalse(deps["needs_patch"])

    def test_cli_limit_overrides_runtime_eval_max_samples(self) -> None:
        runtime_eval_cfg = {"max_eval_samples": 200}
        self.assertEqual(eval_cmrc_strict_module._resolve_max_eval_samples(1, runtime_eval_cfg), 1)

    def test_runtime_session_close_drops_model_references(self) -> None:
        session = TransformersRuntimeSession.__new__(TransformersRuntimeSession)
        session.model = object()
        session.layers = [object()]
        session.tokenizer = object()
        session.rope_spec = object()
        session.close()
        self.assertIsNone(session.model)
        self.assertEqual(session.layers, [])
        self.assertIsNone(session.tokenizer)
        self.assertIsNone(session.rope_spec)


if __name__ == "__main__":
    unittest.main()
