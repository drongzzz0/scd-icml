#!/usr/bin/env python3
"""Apply the SCDConnectorV1 upstream PR files onto a clean vLLM checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


FACTORY_SNIPPET = """KVConnectorFactory.register_connector(
    "SCDConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.scd_connector",
    "SCDConnectorV1",
)
"""

DOC_SNIPPET = """
- **SCDConnectorV1**: refer to [examples/disaggregated/scd_connector/cross_model_kv_handoff.py](../../examples/disaggregated/scd_connector/cross_model_kv_handoff.py) for cross-model Semantic Cache Distillation (SCD) KV cache transfer via `scd-icml`.

  ```bash
  --kv-transfer-config '{"kv_connector":"SCDConnectorV1","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"file","store_root":"/path/to/store","scd_profile":{}}}'
  ```
"""


def apply_bundle(vllm_root: Path, bundle_root: Path) -> list[str]:
    modified: list[str] = []
    files_root = bundle_root / "files"
    for src in sorted(files_root.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(files_root)
        dst = vllm_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        modified.append(rel.as_posix())

    factory_path = vllm_root / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    factory_text = factory_path.read_text(encoding="utf-8")
    if '"SCDConnectorV1"' not in factory_text:
        if not factory_text.endswith("\n"):
            factory_text += "\n"
        factory_text += "\n" + FACTORY_SNIPPET
        factory_path.write_text(factory_text, encoding="utf-8")
        modified.append("vllm/distributed/kv_transfer/kv_connector/factory.py")

    doc_path = vllm_root / "docs/features/disagg_prefill.md"
    if doc_path.is_file():
        doc_text = doc_path.read_text(encoding="utf-8")
        if "**SCDConnectorV1**" not in doc_text:
            anchor = '--kv-transfer-config \'{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}\'\n  ```'
            if anchor in doc_text:
                doc_text = doc_text.replace(anchor, anchor + "\n" + DOC_SNIPPET.rstrip() + "\n", 1)
                doc_text = doc_text.replace(
                    "Now supports 9 types of connectors:",
                    "Now supports 10 types of connectors:",
                    1,
                )
            else:
                doc_text = doc_text.rstrip() + "\n" + DOC_SNIPPET
            doc_path.write_text(doc_text, encoding="utf-8")
            modified.append("docs/features/disagg_prefill.md")

    return modified


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply SCDConnectorV1 PR files to a vLLM git checkout.")
    parser.add_argument("--vllm-root", required=True, type=Path, help="Path to clean vLLM git repository")
    args = parser.parse_args()
    bundle_root = Path(__file__).resolve().parent
    touched = apply_bundle(args.vllm_root.resolve(), bundle_root)
    for item in touched:
        print(f"updated: {item}")


if __name__ == "__main__":
    main()
