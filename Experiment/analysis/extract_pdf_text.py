#!/usr/bin/env python3
"""Extract text and basic metadata from a PDF into project-local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and write plain-text artifacts."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "--output-txt",
        type=Path,
        required=True,
        help="Path to the extracted plain-text output file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to the metadata JSON output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    txt_path = args.output_txt.resolve()
    json_path = args.output_json.resolve()

    reader = PdfReader(str(pdf_path))
    page_texts = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_texts.append(f"===== PAGE {page_index} =====\n{text.strip()}\n")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text("\n".join(page_texts), encoding="utf-8")

    metadata = {
        "source_pdf": str(pdf_path),
        "page_count": len(reader.pages),
        "document_metadata": {k: str(v) for k, v in (reader.metadata or {}).items()},
        "text_output": str(txt_path),
        "extraction_note": "Text was extracted with pypdf page.extract_text().",
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
