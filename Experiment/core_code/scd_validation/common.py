from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _serialize_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def write_json(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=True, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_cell(row.get(key)) for key in fieldnames})
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    sample_means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        sample_means.append(total / n)
    sample_means.sort()
    low_idx = int(0.025 * (n_boot - 1))
    high_idx = int(0.975 * (n_boot - 1))
    return sample_means[low_idx], sample_means[high_idx]
