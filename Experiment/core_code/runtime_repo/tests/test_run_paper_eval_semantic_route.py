from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_paper_eval as run_paper_eval_module  # noqa: E402


class RunPaperEvalSemanticRouteTests(unittest.TestCase):
    def test_merge_semantic_patch_metrics_overrides_patch_errors(self) -> None:
        tmp_root = REPO_ROOT / ".tmp_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        root = tmp_root / f"route_merge_{uuid.uuid4().hex}"
        patch_dir = root / "patch"
        patch_dir.mkdir(parents=True, exist_ok=True)
        try:
            profile_path = root / "profile.json"
            out_path = root / "profile_route.json"

            profile_path.write_text(
                json.dumps(
                    {
                        "patch_fit_mode": "direct_kv_v1",
                        "patch_metric": "patch_kv_error",
                        "layers": [
                            {"layer": 0, "reuse_error": 0.8, "patch_error": 10.0, "patch_k_error": 10.0, "patch_v_error": 10.0, "patch_kv_error": 10.0, "expected_gain": 0.0},
                            {"layer": 1, "reuse_error": 0.2, "patch_error": 20.0, "patch_k_error": 20.0, "patch_v_error": 20.0, "patch_kv_error": 20.0, "expected_gain": 0.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (patch_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "fit_mode": "xnorm_semantic_v1",
                        "layers": {
                            "0": {"k_mse": 0.4, "v_mse": 0.2},
                            "1": {"k_mse": 0.3, "v_mse": 0.1},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_paper_eval_module._merge_patch_metrics_into_profile(profile_path, patch_dir, out_path)
            self.assertEqual(result, out_path)
            merged = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(merged["patch_fit_mode"], "xnorm_semantic_v1")
            self.assertAlmostEqual(merged["layers"][0]["patch_kv_error"], 0.3)
            self.assertAlmostEqual(merged["layers"][0]["expected_gain"], 0.5)
            self.assertAlmostEqual(merged["layers"][1]["patch_kv_error"], 0.2)
            self.assertAlmostEqual(merged["layers"][1]["expected_gain"], 0.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_merge_shared_residual_patch_metrics_overrides_patch_errors(self) -> None:
        tmp_root = REPO_ROOT / ".tmp_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        root = tmp_root / f"route_merge_shared_{uuid.uuid4().hex}"
        patch_dir = root / "patch"
        patch_dir.mkdir(parents=True, exist_ok=True)
        try:
            profile_path = root / "profile.json"
            out_path = root / "profile_route.json"

            profile_path.write_text(
                json.dumps(
                    {
                        "patch_fit_mode": "direct_kv_v1",
                        "patch_metric": "patch_kv_error",
                        "layers": [
                            {"layer": 0, "reuse_error": 0.8, "patch_error": 10.0, "patch_k_error": 10.0, "patch_v_error": 10.0, "patch_kv_error": 10.0, "expected_gain": 0.0},
                            {"layer": 1, "reuse_error": 0.2, "patch_error": 20.0, "patch_k_error": 20.0, "patch_v_error": 20.0, "patch_kv_error": 20.0, "expected_gain": 0.0},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (patch_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "fit_mode": "xnorm_nativekv_shared_residual_mlp_v1",
                        "layers": {
                            "0": {"k_rel_mse": 0.4, "v_rel_mse": 0.2},
                            "1": {"k_rel_mse": 0.3, "v_rel_mse": 0.1},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = run_paper_eval_module._merge_patch_metrics_into_profile(profile_path, patch_dir, out_path)
            self.assertEqual(result, out_path)
            merged = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(merged["patch_fit_mode"], "xnorm_nativekv_shared_residual_mlp_v1")
            self.assertAlmostEqual(merged["layers"][0]["patch_kv_error"], 0.3)
            self.assertAlmostEqual(merged["layers"][0]["expected_gain"], 0.5)
            self.assertAlmostEqual(merged["layers"][1]["patch_kv_error"], 0.2)
            self.assertAlmostEqual(merged["layers"][1]["expected_gain"], 0.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
