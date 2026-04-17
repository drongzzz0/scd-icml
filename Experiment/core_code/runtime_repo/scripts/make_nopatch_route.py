from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a route.json into a no-patch reuse+fallback route.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    route = json.loads(src.read_text(encoding="utf-8"))
    route["fallback_layers"] = sorted(set(route.get("fallback_layers", [])) | set(route.get("patch_layers", [])))
    route["patch_layers"] = []
    route["route_variant"] = "nopatch"
    out.write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
