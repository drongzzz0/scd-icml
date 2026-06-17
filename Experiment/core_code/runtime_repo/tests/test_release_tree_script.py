from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReleaseTreeScriptTest(unittest.TestCase):
    def test_release_tree_includes_vllm_plugin_surface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "release"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_release_tree.py",
                    "--out",
                    str(out_dir),
                    "--force",
                ],
                cwd=root,
                check=True,
            )

            required = [
                "docs/scd_vllm_plugin_mvp.md",
                "configs/scd_vllm_mvp.yaml",
                "reports/scd_vllm_repo_audit.md",
                "reports/scd_vllm_status.md",
                "src/scd_vllm/bundle.py",
                "src/scd_vllm/env_probe.py",
                "src/scd_vllm/injection.py",
                "src/scd_vllm/repo_audit.py",
                "src/scd_vllm/reconstructed_export.py",
                "src/scd_vllm/truekv_oracle.py",
                "src/scd_vllm_plugin/cli.py",
                "src/scd_vllm_plugin/engine.py",
                "scripts/scd_vllm/run_00_repo_audit.sh",
                "scripts/scd_vllm/run_01_env_probe.sh",
                "scripts/scd_vllm/run_02_hf_export.sh",
                "scripts/scd_vllm/run_03_vllm_baseline.sh",
                "scripts/scd_vllm/run_04_truekv_oracle.sh",
                "scripts/scd_vllm/run_05_reconstructed_export.sh",
                "scripts/scd_vllm/run_05_scd_injection.sh",
                "scripts/scd_vllm/run_06_report.sh",
            ]
            for rel in required:
                self.assertTrue((out_dir / rel).is_file(), rel)

            self.assertFalse(any(out_dir.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
