from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WIKITEXT_PROMPT_TEMPLATE = "Continue the text:\n{prefix}"


@dataclass
class WikiText2Sample:
    qid: str
    prompt: str
    answers: list[str]
    source_text: str


def build_wikitext2_prompt(prefix: str) -> str:
    return WIKITEXT_PROMPT_TEMPLATE.format(prefix=prefix.strip())


def _candidate_files(root: Path) -> list[Path]:
    preferred = [
        root / "wiki.test.txt",
        root / "wiki.test.tokens",
        root / "test.txt",
        root / "test.tokens",
        root / "wiki.valid.txt",
        root / "wiki.valid.tokens",
    ]
    out: list[Path] = [path for path in preferred if path.exists()]
    if out:
        return out
    extras: list[Path] = []
    for path in root.rglob("*.txt"):
        extras.append(path)
    for path in root.rglob("*.tokens"):
        extras.append(path)
    return sorted(extras)


def _normalize_wikitext_split(split: str | None) -> str:
    text = str(split or "test").strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "valid": "valid",
        "validation": "valid",
        "dev": "valid",
        "test": "test",
        "all": "all",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(f"unsupported WikiText-2 split: {split}")
    return normalized


def _split_candidate_files(root: Path, split: str) -> list[Path]:
    split_key = _normalize_wikitext_split(split)
    if split_key == "all":
        files: list[Path] = []
        for path in root.rglob("*.txt"):
            files.append(path)
        for path in root.rglob("*.tokens"):
            files.append(path)
        return sorted(files)
    preferred_map = {
        "train": [
            root / "wiki.train.txt",
            root / "wiki.train.tokens",
            root / "train.txt",
            root / "train.tokens",
        ],
        "valid": [
            root / "wiki.valid.txt",
            root / "wiki.valid.tokens",
            root / "valid.txt",
            root / "valid.tokens",
            root / "validation.txt",
            root / "validation.tokens",
            root / "dev.txt",
            root / "dev.tokens",
        ],
        "test": [
            root / "wiki.test.txt",
            root / "wiki.test.tokens",
            root / "test.txt",
            root / "test.tokens",
        ],
    }
    return [path for path in preferred_map[split_key] if path.exists()]


def _normalize_wikitext_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def wikitext2_em(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_wikitext_answer(prediction) == _normalize_wikitext_answer(ground_truth) else 0.0


def wikitext2_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_wikitext_answer(prediction).split()
    gt_tokens = _normalize_wikitext_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in pred_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in gt_tokens:
        value = counts.get(token, 0)
        if value > 0:
            overlap += 1
            counts[token] = value - 1
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_wikitext2_prediction(prediction: str, ground_truths: Iterable[str]) -> dict[str, float | str]:
    refs = [str(x).strip() for x in ground_truths if str(x).strip()]
    if not refs:
        return {"f1": 0.0, "em": 0.0, "best_answer": ""}
    best_f1 = 0.0
    best_em = 0.0
    best_answer = refs[0]
    for answer in refs:
        f1 = wikitext2_f1(prediction, answer)
        em = wikitext2_em(prediction, answer)
        if (f1, em) > (best_f1, best_em):
            best_f1 = f1
            best_em = em
            best_answer = answer
    return {"f1": float(best_f1), "em": float(best_em), "best_answer": best_answer}


def load_wikitext2_samples(
    data_root: str | Path,
    limit: int | None = None,
    split: str | None = "test",
) -> list[WikiText2Sample]:
    root = Path(data_root)
    candidates = [root / "wikitext", root / "WikiText-2", root / "wikitext-2"]
    split_key = _normalize_wikitext_split(split)
    saw_existing_candidate = False
    samples: list[WikiText2Sample] = []
    for candidate_root in candidates:
        if not candidate_root.exists():
            continue
        saw_existing_candidate = True
        for path in _split_candidate_files(candidate_root, split_key):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_idx, line in enumerate(text.splitlines()):
                line = re.sub(r"\s+", " ", line.strip())
                if len(line) < 80:
                    continue
                words = line.split()
                if len(words) < 16:
                    continue
                prefix_count = min(max(8, len(words) // 2), 48)
                target_words = words[prefix_count : prefix_count + 32]
                if not target_words:
                    continue
                prefix = " ".join(words[:prefix_count]).strip()
                target = " ".join(target_words).strip()
                if not prefix or not target:
                    continue
                samples.append(
                    WikiText2Sample(
                        qid=f"{path.stem}_{line_idx}",
                        prompt=build_wikitext2_prompt(prefix),
                        answers=[target],
                        source_text=line,
                    )
                )
                if limit is not None and len(samples) >= int(limit):
                    return samples
        if samples:
            return samples
    if saw_existing_candidate:
        raise ValueError(f"found WikiText-2 directory but no usable split={split_key} text files: {candidates}")
    raise FileNotFoundError(f"missing usable WikiText-2 data directory: {candidates}")
