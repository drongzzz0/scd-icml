from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


class DefaultStrictContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.config_root = cls.repo_root / "configs"

    def _load_json(self, *parts: str) -> dict:
        path = self.config_root.joinpath(*parts)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _load_module(self, relative_path: str, module_name: str):
        path = self.repo_root / relative_path
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        old_argv = sys.argv
        try:
            sys.argv = [str(path)]
            spec.loader.exec_module(module)
        finally:
            sys.argv = old_argv
        return module

    def test_train_yaml_uses_canonical_strict_defaults(self) -> None:
        cfg = self._load_json("train.yaml")
        self.assertEqual(cfg["xkv_group"]["group_size"], 1)
        self.assertEqual(cfg["patch"]["fit_mode"], "xnorm_nativekv_shared_residual_mlp_v1")
        self.assertEqual(int(cfg["patch"]["patch_budget"]), 32)

    def test_eval_yaml_disables_archived_runtime_methods(self) -> None:
        cfg = self._load_json("eval.yaml")
        runtime = cfg["runtime_eval"]
        self.assertTrue(runtime["run_full_consumer_baseline"])
        self.assertTrue(runtime["run_reuse_only"])
        self.assertFalse(runtime["run_reuse_fallback"])
        self.assertFalse(runtime["run_reuse_patch"])
        self.assertFalse(runtime["run_reuse_patch_recompute"])
        self.assertFalse(runtime["run_patch_only"])

    def test_route_template_defaults_to_all_layer_patch_only(self) -> None:
        route = self._load_json("route_patch_only_all32.json")
        self.assertEqual(route["patch_layers"], list(range(32)))
        self.assertEqual(route["selected_recompute_layers"], [])
        self.assertEqual(route["fallback_layers"], [])

    def test_paper_runner_defaults_point_to_legacy_configs(self) -> None:
        module = self._load_module("scripts/run_paper_eval.py", "run_paper_eval_test")
        parser = module.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.train_config, "configs/train_paper_semantic_legacy.json")
        self.assertEqual(args.eval_config, "configs/eval_paper_semantic_legacy.json")

    def test_default_strict_runner_points_to_mistrallite_to_mistral7(self) -> None:
        module = self._load_module("src/scd_repro/cli/run_mistral_strict_default.py", "run_mistral_strict_default_test")
        parser = module.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.models_config, "configs/models_ser_mistrallite_to_mistral7_20260405.json")
        self.assertEqual(args.train_config, "configs/train.yaml")
        self.assertEqual(args.eval_config, "configs/eval.yaml")
        self.assertEqual(args.patch_eval_config, "configs/eval_patch_only.yaml")
        self.assertEqual(args.route_template, "configs/route_patch_only_all32.json")
        self.assertEqual(args.samples, 200)
        self.assertEqual(args.eval_samples, 200)
        self.assertEqual(args.eval_offset, -1)

    def test_prepare_cmrc_split_uses_disjoint_holdout_by_default(self) -> None:
        module = self._load_module("src/scd_repro/cli/run_mistral_strict_default.py", "run_mistral_strict_default_split_test")
        fake_samples = [SimpleNamespace(qid=f"q{i}", prompt=f"p{i}") for i in range(10)]

        original_loader = module.load_cmrc2018_samples
        try:
            module.load_cmrc2018_samples = lambda data_root, limit=None: fake_samples[:limit]
            train_rows, eval_rows, load_limit = module._prepare_cmrc_split(
                Path("unused"),
                train_offset=0,
                train_samples=4,
                eval_offset=-1,
                eval_samples=3,
            )
        finally:
            module.load_cmrc2018_samples = original_loader

        self.assertEqual(load_limit, 7)
        self.assertEqual([row.qid for row in train_rows], ["q0", "q1", "q2", "q3"])
        self.assertEqual([row.qid for row in eval_rows], ["q4", "q5", "q6"])


if __name__ == "__main__":
    unittest.main()
