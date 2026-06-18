from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SCDVLLMOfficialLaunchExampleTest(unittest.TestCase):
    def test_dry_run_validates_launch_spec_without_vllm(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            launch_spec = tmp / "launch_spec.json"
            out = tmp / "dry_run.json"
            payload = {
                "bundle_dir": str(tmp / "bundle"),
                "storage_dir": str(tmp / "storage"),
                "bundle": {
                    "route_id": "route-a",
                    "route_hash": "abc",
                    "token_hash": "def",
                },
                "storage": {
                    "exists": True,
                    "safetensor_files": 1,
                },
                "safety": {
                    "multi_tenant_safe": False,
                    "dynamic_request_routing": False,
                    "trusted_bundle_required": True,
                },
                "environment": {
                    "SCD_PRELOADED_STATS_DIR": str(tmp / "stats"),
                },
                "llm_kwargs": {
                    "model": "consumer",
                    "tokenizer": "tok",
                    "disable_hybrid_kv_cache_manager": True,
                    "kv_transfer_config": {
                        "kv_connector": "SCDPreloadedConnector",
                        "kv_connector_module_path": "scd_vllm.preloaded_connector",
                    },
                },
            }
            launch_spec.write_text(json.dumps(payload), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "examples/scd_vllm_official_launch.py",
                    "--launch-spec",
                    str(launch_spec),
                    "--dry-run",
                    "--out",
                    str(out),
                ],
                cwd=root,
                check=True,
            )

            metrics = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(metrics["mode"], "scd_vllm_official_launch_example")
            self.assertTrue(metrics["dry_run"])
            self.assertTrue(metrics["ready"])
            self.assertEqual(metrics["missing"], [])
            self.assertEqual(metrics["kv_connector"], "SCDPreloadedConnector")
            self.assertEqual(metrics["route_id"], "route-a")


if __name__ == "__main__":
    unittest.main()
