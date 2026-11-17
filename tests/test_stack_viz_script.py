from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pytest

pytest.importorskip("matplotlib")


class StackVizScriptTest(unittest.TestCase):
    def test_cli_writes_pdf_png_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist_csv = root / "dist.csv"
            dist_meta = root / "dist_meta.json"
            tsne_csv = root / "tsne.csv"
            tsne_meta = root / "tsne_meta.json"
            out_dir = root / "out"

            dist_csv.write_text(
                "\n".join(
                    [
                        "layer,point_id,source_rel_l2,translated_rel_l2",
                        "24,0,0.80,0.50",
                        "24,1,0.70,0.45",
                        "24,2,0.75,0.40",
                        "24,3,0.65,0.35",
                    ]
                ),
                encoding="utf-8",
            )
            dist_meta.write_text(json.dumps({"median_rel_l2_reduction": 0.4}), encoding="utf-8")
            tsne_rows = ["layer,point_id,series,x,y"]
            for series, offset in (("source", 0.0), ("target", 2.0), ("translated_reuse", 1.0)):
                for idx in range(4):
                    tsne_rows.append(f"24,{idx},{series},{offset + idx * 0.1},{offset + idx * 0.2}")
            tsne_csv.write_text("\n".join(tsne_rows), encoding="utf-8")
            tsne_meta.write_text(json.dumps({"layer": 24, "perplexity": 5}), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_stack_viz.py",
                    "--distance-csv",
                    str(dist_csv),
                    "--distance-meta",
                    str(dist_meta),
                    "--tsne-csv",
                    str(tsne_csv),
                    "--tsne-meta",
                    str(tsne_meta),
                    "--out-dir",
                    str(out_dir),
                    "--basename",
                    "stack_viz_test",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            for suffix in ("pdf", "png", "svg"):
                path = out_dir / f"stack_viz_test.{suffix}"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
