from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CMRC_PROMPT_TEMPLATE = "阅读以下材料并回答问题。\n材料：{context}\n问题：{question}\n答案："

_PUNCT = {
    "-", ":", "_", "*", "^", "/", "\\", "~", "`", "+", "=",
    "，", "。", "：", "？", "！", "“", "”", "；", "’", "《", "》", "……", "·",
    "、", "「", "」", "（", "）", "－", "～", "『", "』", "|",
    ",", ".", "?", "!", '"', "'", ";", "(", ")", "[", "]", "{", "}", "<", ">",
}


@dataclass
class CMRCSample:
    qid: str
    context: str
    question: str
    answers: list[str]
    prompt: str


def _load_structured_file(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return rows
        raise


def build_cmrc_prompt(context: str, question: str) -> str:
    return CMRC_PROMPT_TEMPLATE.format(context=context.strip(), question=question.strip())


def _collect_answer_texts(raw_answers) -> list[str]:
    answers: list[str] = []
    if isinstance(raw_answers, list):
        for row in raw_answers:
            if isinstance(row, dict):
                text = str((row or {}).get("text", "")).strip()
            else:
                text = str(row or "").strip()
            if text and text not in answers:
                answers.append(text)
    return answers


def load_cmrc2018_samples(data_root: str | Path, limit: int | None = None) -> list[CMRCSample]:
    root = Path(data_root)
    candidates = [
        root / "cmrc2018" / "cmrc2018_dev.json",
        root / "prompts" / "cmrc2018_dev.json",
        root / "cmrc2018" / "cmrc2018_train.json",
    ]
    saw_existing_candidate = False
    for path in candidates:
        if not path.exists():
            continue
        saw_existing_candidate = True
        obj = _load_structured_file(path)
        samples: list[CMRCSample] = []
        data = obj.get("data") if isinstance(obj, dict) else None
        if isinstance(data, list):
            for article in data:
                for para in article.get("paragraphs", []):
                    context = str(para.get("context", "")).strip()
                    if not context:
                        continue
                    for qa in para.get("qas", []):
                        question = str(qa.get("question", "")).strip()
                        if not question:
                            continue
                        answers = _collect_answer_texts(qa.get("answers", []))
                        if not answers:
                            continue
                        qid = str(qa.get("id", f"cmrc_{len(samples)}"))
                        samples.append(
                            CMRCSample(
                                qid=qid,
                                context=context,
                                question=question,
                                answers=answers,
                                prompt=build_cmrc_prompt(context, question),
                            )
                        )
                        if limit is not None and len(samples) >= limit:
                            return samples
        elif isinstance(obj, list):
            for row in obj:
                context = str((row or {}).get("context_text", "")).strip()
                if not context:
                    continue
                for qa in (row or {}).get("qas", []):
                    question = str((qa or {}).get("query_text", "")).strip()
                    if not question:
                        continue
                    answers = _collect_answer_texts((qa or {}).get("answers", []))
                    if not answers:
                        continue
                    qid = str((qa or {}).get("query_id", f"cmrc_{len(samples)}"))
                    samples.append(
                        CMRCSample(
                            qid=qid,
                            context=context,
                            question=question,
                            answers=answers,
                            prompt=build_cmrc_prompt(context, question),
                        )
                    )
                    if limit is not None and len(samples) >= limit:
                        return samples
        if samples:
            return samples
    if saw_existing_candidate:
        raise ValueError(f"找到 CMRC 文件但未识别其结构: {candidates}")
    raise FileNotFoundError(f"未找到可用的 CMRC2018 dev 数据: {candidates}")


def _mixed_segmentation(text: str, remove_punctuation: bool) -> list[str]:
    text = str(text or "").strip().lower()
    tokens: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+|[^\w\s]", text, flags=re.IGNORECASE):
        if remove_punctuation and token in _PUNCT:
            continue
        if token.strip():
            tokens.append(token)
    return tokens


def normalize_cmrc_answer(text: str) -> str:
    return "".join(_mixed_segmentation(text, remove_punctuation=True))


def cmrc_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _mixed_segmentation(prediction, remove_punctuation=True)
    gt_tokens = _mixed_segmentation(ground_truth, remove_punctuation=True)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    counter: dict[str, int] = {}
    for token in pred_tokens:
        counter[token] = counter.get(token, 0) + 1

    overlap = 0
    for token in gt_tokens:
        count = counter.get(token, 0)
        if count > 0:
            overlap += 1
            counter[token] = count - 1

    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def cmrc_em(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_cmrc_answer(prediction) == normalize_cmrc_answer(ground_truth) else 0.0


def score_cmrc_prediction(prediction: str, ground_truths: Iterable[str]) -> dict[str, float | str]:
    refs = [str(x).strip() for x in ground_truths if str(x).strip()]
    if not refs:
        return {"f1": 0.0, "em": 0.0, "best_answer": ""}

    best_f1 = 0.0
    best_em = 0.0
    best_answer = refs[0]
    for answer in refs:
        f1 = cmrc_f1(prediction, answer)
        em = cmrc_em(prediction, answer)
        if (f1, em) > (best_f1, best_em):
            best_f1 = f1
            best_em = em
            best_answer = answer
    return {"f1": float(best_f1), "em": float(best_em), "best_answer": best_answer}


def extract_answer_text(text: str) -> str:
    out = str(text or "").strip()
    for prefix in ["答案：", "答案:", "answer:", "Answer:"]:
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
    for sep in ["\n", "</s>", "<|endoftext|>"]:
        if sep in out:
            out = out.split(sep, 1)[0].strip()
    return out.strip()
