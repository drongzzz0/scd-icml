#!/usr/bin/env python3
"""Score saved CMRC2018 dev predictions using the authors' evaluation script."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_COMMIT = "c0eb1b6ba219847457e6af3180da722bbeb656af"
SOURCE_ROOT = f"https://raw.githubusercontent.com/ymcui/cmrc2018/{SOURCE_COMMIT}"
DEV_URL = f"{SOURCE_ROOT}/squad-style-data/cmrc2018_dev.json"
SCORER_URL = f"{SOURCE_ROOT}/baseline/cmrc2018_evaluate.py"
DEV_SHA256 = "e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f"
SCORER_SHA256 = "c14090c77e22531e58b8170952b74d6ebe6e11f4a605e7feba6e56a7caa2735c"
DEV_N = 3219


def _read_pinned(path: Path, label: str, expected_sha256: str) -> bytes:
    contents = path.read_bytes()
    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual_sha256} (expected {expected_sha256})")
    return contents


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_predictions(path: Path, qids: set[str]) -> dict[str, str]:
    predictions = json.loads(path.read_bytes(), object_pairs_hook=_unique_json_object)
    if not isinstance(predictions, dict) or any(not isinstance(value, str) for value in predictions.values()):
        raise ValueError(f"{path} must map official qids to prediction strings")
    unknown = predictions.keys() - qids
    if unknown:
        raise ValueError(f"{path} contains unknown CMRC2018 qid(s): {sorted(unknown)[:5]}")
    return predictions


def _score(evaluate, dataset: dict, predictions: dict[str, str]) -> dict[str, float | int]:
    f1, em, n, skipped = evaluate(dataset, predictions)
    if n != DEV_N or skipped != n - len(predictions):
        raise ValueError("official scorer returned unexpected question counts")
    return {"F1": f1, "EM": em, "N": n, "SKIP": skipped}


def _load_evaluator(path):
    scorer = _read_pinned(path, "scorer", SCORER_SHA256)
    # Execute the verified bytes without invoking the upstream CLI, which rounds its scores.
    namespace = {"__name__": "cmrc2018_official_evaluator"}
    exec(compile(scorer, str(path), "exec"), namespace)
    return namespace["evaluate"], {
        "source": SCORER_URL, "sha256": SCORER_SHA256, "nltk_version": namespace["nltk"].__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help=f"unmodified official dev file from {DEV_URL}")
    parser.add_argument("--scorer", type=Path, required=True, help=f"unmodified official Python 3 scorer from {SCORER_URL}")
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--scd-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="also write the JSON report to this path")
    args = parser.parse_args(argv)

    try:
        dataset = json.loads(_read_pinned(args.dataset, "dataset", DEV_SHA256))
        ids = [qa["id"] for article in dataset["data"] for para in article["paragraphs"] for qa in para["qas"]]
        qids = set(ids)
        if len(ids) != DEV_N or len(qids) != DEV_N:
            raise ValueError(f"official dev must contain {DEV_N} unique questions")

        native_predictions = _read_predictions(args.native_predictions, qids)
        scd_predictions = _read_predictions(args.scd_predictions, qids)

        evaluate, scorer = _load_evaluator(args.scorer)
        native = _score(evaluate, dataset, native_predictions)
        scd = _score(evaluate, dataset, scd_predictions)
        report = {
            "dataset": {"source": DEV_URL, "sha256": DEV_SHA256},
            "scorer": scorer,
            "native": native,
            "scd": scd,
            "delta": {"F1": scd["F1"] - native["F1"], "EM": scd["EM"] - native["EM"]},
        }
        output = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.write_text(output, encoding="utf-8")
    except ImportError as exc:
        parser.error(f"the official scorer requires NLTK: {exc}")
    except LookupError as exc:
        parser.error(f"the official scorer requires the NLTK English tokenizer data (punkt_tab for NLTK 3.9): {exc}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
