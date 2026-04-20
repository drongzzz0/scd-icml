from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scd_repro.patch import PATCH_FIT_MODE_XNORM_SEMANTIC_V1, XNormSemanticPatchLayer, pack_xnorm_semantic_patch_layer  # noqa: E402
import scd_repro.pipeline as pipeline_module  # noqa: E402
from scd_repro.pipeline import ConsumerSemanticRuntimeLayer, _get_semantic_patch_layer, _prepare_cache_for_method  # noqa: E402
from scd_repro.torch_utils import require_torch  # noqa: E402


class SemanticPatchPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = require_torch()

    def _build_patch_layer(self) -> XNormSemanticPatchLayer:
        module = XNormSemanticPatchLayer(
            hidden_dim=4,
            code_dim=4,
            mlp_hidden_dim=4,
            enc_h_is_identity=True,
        ).to(device="cpu", dtype=self.torch.float32)
        with self.torch.no_grad():
            module.patch_mlp.fc1.weight.zero_()
            module.patch_mlp.fc1.bias.zero_()
            module.patch_mlp.fc2.weight.zero_()
            module.patch_mlp.fc2.bias.copy_(self.torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=self.torch.float32))
        module.eval()
        return module

    def _runtime_layer(self) -> ConsumerSemanticRuntimeLayer:
        return ConsumerSemanticRuntimeLayer(
            input_norm=None,
            k_proj_weight=self.torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=self.torch.float32),
            k_proj_bias=None,
            v_proj_weight=self.torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=self.torch.float32),
            v_proj_bias=None,
            k_norm=None,
        )

    def test_prepare_cache_supports_xnorm_semantic_patch(self) -> None:
        patch_layer = self._build_patch_layer()
        patch_meta = {
            "fit_mode": PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
            "selected_layers": [0],
            "patch_apply_mode": "overwrite",
            "__runtime_semantic_patch_layers__": {0: patch_layer},
            "__consumer_semantic_runtime_layers__": {0: self._runtime_layer()},
        }
        meta = {
            "num_layers": 1,
            "batch_size": 1,
            "prefix_tokens": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "hidden_dim": 4,
        }
        arrays = {
            "xnorm_a.0": self.torch.zeros((1, 1, 4), dtype=self.torch.float32),
        }
        route = {"reuse_layers": [], "patch_layers": [0], "fallback_layers": []}

        writer = _prepare_cache_for_method(
            "scd_full",
            arrays,
            meta,
            reuse=None,
            patch_bundle={},
            patch_meta=patch_meta,
            route=route,
            models_config=None,
            infer_device="cpu",
            cpu_fallback=True,
        )

        expected_k = self.torch.tensor([[[[1.0, 2.0]]]], dtype=self.torch.float32)
        expected_v = self.torch.tensor([[[[3.0, 4.0]]]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(writer.k_cache[0], expected_k))
        self.assertTrue(self.torch.allclose(writer.v_cache[0], expected_v))

    def test_get_semantic_patch_layer_loads_from_shard_when_bundle_empty(self) -> None:
        patch_layer = self._build_patch_layer()
        arrays = pack_xnorm_semantic_patch_layer(0, patch_layer)
        patch_meta = {
            "fit_mode": PATCH_FIT_MODE_XNORM_SEMANTIC_V1,
            "hidden_dim": 4,
            "code_dim": 4,
            "mlp_hidden_dim": 4,
            "enc_h_is_identity": True,
            "shard_manifest": {
                "layer_to_shard": {"0": "layers_0000_0000.pt"},
                "layer_keys": {"0": list(arrays.keys())},
            },
            "__patch_ckpt_root__": str(REPO_ROOT / ".tmp_tests"),
        }
        with mock.patch.object(pipeline_module, "load_bundle", return_value=SimpleNamespace(arrays=arrays)):
            restored = _get_semantic_patch_layer({}, patch_meta, layer=0, infer_device="cpu")
            out = restored(self.torch.zeros((1, 4), dtype=self.torch.float32))

        expected = self.torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=self.torch.float32)
        self.assertTrue(self.torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
