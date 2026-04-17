from __future__ import annotations

import argparse

from ..validate import validate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 SCD 产物目录或文件路径")
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    kind = validate_path(args.path)
    print(f"校验通过: {kind}")


if __name__ == "__main__":
    main()

