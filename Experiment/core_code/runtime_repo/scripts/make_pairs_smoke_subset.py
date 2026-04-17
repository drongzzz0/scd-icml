from __future__ import annotations

import argparse

from scd_repro.pipeline import iter_pair_chunks, save_chunked_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a small chunked pairs subset for smoke runs.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--keep", type=int, default=16)
    parser.add_argument("--artifact-dtype", default="float32")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    arrays, meta = next(iter(iter_pair_chunks(args.src, as_numpy=True)))
    keep = max(1, min(int(args.keep), int(meta["batch_size"])))
    subset_arrays = {key: value[:keep].copy() for key, value in arrays.items()}
    subset_meta = dict(meta)
    subset_meta["batch_size"] = keep
    subset_meta["eval_samples"] = keep
    save_chunked_pairs(args.dst, [(subset_meta, subset_arrays)], artifact_dtype=str(args.artifact_dtype))


if __name__ == "__main__":
    main()
