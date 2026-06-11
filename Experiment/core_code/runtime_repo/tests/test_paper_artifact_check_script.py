from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PaperArtifactCheckScriptTest(unittest.TestCase):
    def test_cli_validates_figures_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            bundle_dir = root / "bundle"
            (paper_dir / "generated_figures").mkdir(parents=True)
            (bundle_dir / "figures").mkdir(parents=True)
            tex = paper_dir / "paper.tex"
            tex.write_text(
                r"""
\begin{figure}
\includegraphics[width=0.9\linewidth]{generated_figures/layerwise_xnorm_replacement.pdf}
\end{figure}
""",
                encoding="utf-8",
            )
            (paper_dir / "generated_figures" / "layerwise_xnorm_replacement.pdf").write_text("%PDF\n", encoding="utf-8")
            (bundle_dir / "figures" / "layerwise_xnorm_replacement.pdf").write_text("%PDF\n", encoding="utf-8")
            (bundle_dir / "paper_artifact_manifest.json").write_text(
                json.dumps({"status": "ok", "missing": []}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_paper_artifacts.py",
                    "--paper-tex",
                    str(tex),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--forbid-image",
                    "patch_error1.pdf",
                    "--require-image",
                    "generated_figures/layerwise_xnorm_replacement.pdf",
                    "--require-bundle-file",
                    "figures/layerwise_xnorm_replacement.pdf",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("paper_artifacts_ok", completed.stdout)

    def test_cli_rejects_forbidden_or_missing_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            paper_dir.mkdir(parents=True)
            tex = paper_dir / "paper.tex"
            tex.write_text(r"\includegraphics{patch_error1.pdf}", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_paper_artifacts.py",
                    "--paper-tex",
                    str(tex),
                    "--forbid-image",
                    "patch_error1.pdf",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("forbidden image referenced", completed.stdout)

    def test_cli_checks_required_and_forbidden_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            paper_dir.mkdir(parents=True)
            tex = paper_dir / "paper.tex"
            tex.write_text(
                "Verified release evidence. Earlier rows are not claim-facing.",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_paper_artifacts.py",
                    "--paper-tex",
                    str(tex),
                    "--require-text",
                    "not claim-facing",
                    "--forbid-text",
                    "unsupported baseline table",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

            failed = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_paper_artifacts.py",
                    "--paper-tex",
                    str(tex),
                    "--forbid-text",
                    "not claim-facing",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("forbidden text present", failed.stdout)


if __name__ == "__main__":
    unittest.main()
