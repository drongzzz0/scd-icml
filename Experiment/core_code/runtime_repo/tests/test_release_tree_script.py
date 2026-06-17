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
                "src/scd_vllm/bundle.py",
                "src/scd_vllm/injection.py",
                "src/scd_vllm/reconstructed_export.py",
                "src/scd_vllm_plugin/cli.py",
                "src/scd_vllm_plugin/engine.py",
            ]
            for rel in required:
                self.assertTrue((out_dir / rel).is_file(), rel)

            self.assertFalse(any(out_dir.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
