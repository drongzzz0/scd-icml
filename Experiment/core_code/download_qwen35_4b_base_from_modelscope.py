from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-4B-Base from ModelScope into the fixed SCD validation path."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-4B-Base",
        help="ModelScope model id. Defaults to the primary strict-Qwen3.5 producer checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="/disk_n/c101-3-1/models/Qwen3.5-4B-Base",
        help="Destination directory on the remote host.",
    )
    return parser.parse_args()


def verify_download(output_dir: Path) -> dict:
    config_candidates = [
        output_dir / "config.json",
        output_dir / "configuration.json",
    ]
    config_path = next((path for path in config_candidates if path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"download completed but no config file was found under {output_dir}"
        )
    config = json.loads(config_path.read_text())
    return {
        "config_path": str(config_path),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except Exception as exc:  # pragma: no cover - runtime dependency on remote host
        raise SystemExit(
            "ModelScope is not available in the current Python environment. "
            "Install it first with `python3 -m pip install -U modelscope`.\n"
            f"Original import error: {exc!r}"
        ) from exc

    model_dir = snapshot_download(args.model_id, local_dir=str(output_dir))
    verified = verify_download(output_dir)

    print(f"downloaded_model_dir={model_dir}")
    print(f"verified_output_dir={output_dir}")
    print(f"verified_config_path={verified['config_path']}")
    print(f"verified_model_type={verified['model_type']}")
    print(f"verified_architectures={verified['architectures']}")


if __name__ == "__main__":
    main()
