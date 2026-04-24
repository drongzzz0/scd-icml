from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HOTPOT_PROMPT_TEMPLATE = (
    "Read the question and answer using only the evidence.\n"
    "Output only the shortest answer.\n"
    "Do not repeat the question or evidence.\n"
    "If the question is yes/no, answer only yes or no.\n"
    "If the answer is a person, place, organization, number, or date, output only that short span.\n"
    "Question: {question}\n"
    "Evidence:\n{context}\n"
    "Answer:"
)
HOTPOT_MAX_EVIDENCE_CHARS = 900
HOTPOT_MAX_PASSAGES = 4
HOTPOT_MAX_SENTENCES_PER_PASSAGE = 3

_QUESTION_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "from",
    "by",
    "with",
    "at",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "which",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "first",
    "second",
    "before",
    "after",
    "into",
    "than",
    "then",
    "between",
    "during",
    "through",
    "about",
    "across",
    "under",
    "over",
    "case",
}

_QUESTION_PHRASE_BLACKLIST = {
    "which",
    "what",
    "when",
    "where",
    "who",
    "how",
    "how many",
    "how much",
    "how long",
    "how old",
    "yes",
    "no",
}

_QUESTION_PHRASE_CONNECTORS = {"of", "the", "and", "for", "in", "on", "at", "de", "la", "le", "du", "van", "von", "v.", "vs."}


@dataclass
class HotpotQASample:
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


def _hotpot_context_to_text(context) -> str:
    if isinstance(context, list):
        parts: list[str] = []
        for block in context:
            if isinstance(block, list) and len(block) >= 2:
                title = str(block[0]).strip()
                sents = block[1]
                if isinstance(sents, list):
                    body = " ".join(str(x).strip() for x in sents if str(x).strip())
                else:
                    body = str(sents).strip()
                if title and body:
                    parts.append(f"[{title}] {body}")
                elif body:
                    parts.append(body)
            else:
                text = str(block).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(context, dict):
        return json.dumps(context, ensure_ascii=False)
    return str(context or "").strip()


def _normalize_question_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 2 and token not in _QUESTION_STOPWORDS
    }


def _numeric_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"\b\d+\b", str(text or ""))}


def _extract_question_phrases(question: str) -> list[str]:
    text = str(question or "").strip()
    patterns = re.finditer(
        r"(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+(?:[A-Z][A-Za-z0-9&.'-]*|v\.|vs\.|of|the|and|for|in|on|at|de|la|le|du|van|von))*",
        text,
    )
    phrases: list[str] = []
    seen: set[str] = set()
    for match in patterns:
        phrase = " ".join(match.group(0).split()).strip(" ,;:.?")
        if not phrase:
            continue
        tokens = phrase.split()
        while tokens and tokens[-1].lower() in _QUESTION_PHRASE_CONNECTORS:
            tokens.pop()
        phrase = " ".join(tokens).strip()
        lowered = phrase.lower()
        if not phrase or lowered in _QUESTION_PHRASE_BLACKLIST:
            continue
        if lowered not in seen:
            seen.add(lowered)
            phrases.append(phrase)
    phrases.sort(key=lambda value: (-len(value.split()), -len(value), value.lower()))
    return phrases[: max(2, HOTPOT_MAX_PASSAGES)]


def _phrase_match_score(text: str, phrase: str) -> float:
    hay = f" {str(text or '').lower()} "
    needle = f" {phrase.lower()} "
    if needle in hay:
        return 1.0
    phrase_terms = [term for term in _normalize_question_terms(phrase) if len(term) >= 2]
    if not phrase_terms:
        return 0.0
    text_terms = _normalize_question_terms(text)
    overlap = len(set(phrase_terms) & text_terms)
    return float(overlap / max(1, len(set(phrase_terms))))


def _sentence_overlap_score(text: str, question_terms: set[str], numeric_terms: set[str], phrases: list[str]) -> float:
    terms = _normalize_question_terms(text)
    overlap = len(question_terms & terms)
    digit_overlap = len(numeric_terms & _numeric_terms(text))
    phrase_bonus = sum(_phrase_match_score(text, phrase) for phrase in phrases)
    return float(overlap * 3 + digit_overlap * 2 + phrase_bonus * 4)


def _ranked_hotpot_passages(context, question: str) -> list[str]:
    question_terms = _normalize_question_terms(question)
    question_digits = _numeric_terms(question)
    question_phrases = _extract_question_phrases(question)
    if not isinstance(context, list):
        flat = _hotpot_context_to_text(context)
        return [flat] if flat else []

    candidates: list[dict[str, object]] = []
    for idx, block in enumerate(context):
        title = ""
        sentences: list[str] = []
        if isinstance(block, list) and len(block) >= 2:
            title = str(block[0]).strip()
            raw_sents = block[1]
            if isinstance(raw_sents, list):
                sentences = [str(x).strip() for x in raw_sents if str(x).strip()]
            else:
                text = str(raw_sents).strip()
                if text:
                    sentences = [text]
        else:
            text = str(block).strip()
            if text:
                sentences = [text]
        if not title and not sentences:
            continue

        title_terms = _normalize_question_terms(title)
        title_phrase_hits = {phrase for phrase in question_phrases if _phrase_match_score(title, phrase) >= 0.99}
        sentence_rows: list[tuple[float, int, str]] = []
        for sent_idx, sent in enumerate(sentences):
            score = _sentence_overlap_score(sent, question_terms, question_digits, question_phrases)
            sentence_rows.append((score, sent_idx, sent))

        sentence_rows.sort(key=lambda row: (-row[0], row[1]))
        chosen = sentence_rows[: max(1, HOTPOT_MAX_SENTENCES_PER_PASSAGE)]
        chosen.sort(key=lambda row: row[1])
        chosen_sentences = [row[2] for row in chosen if row[2].strip()]
        if not chosen_sentences and sentences:
            chosen_sentences = sentences[:HOTPOT_MAX_SENTENCES_PER_PASSAGE]

        merged_text = " ".join(chosen_sentences).strip()
        rendered = f"[{title}] {merged_text}".strip() if title else merged_text
        body_terms = _normalize_question_terms(merged_text)
        body_phrase_hits = {phrase for phrase in question_phrases if _phrase_match_score(merged_text, phrase) >= 0.99}
        coverage_terms = title_terms | body_terms
        title_overlap = len(question_terms & title_terms)
        body_overlap = len(question_terms & body_terms)
        best_sentence = chosen[0][0] if chosen else 0.0
        numeric_overlap = len(question_digits & (_numeric_terms(title) | _numeric_terms(merged_text)))
        base_score = float(
            title_overlap * 5
            + body_overlap * 3
            + best_sentence
            + numeric_overlap * 2
            + len(title_phrase_hits) * 8
            + len(body_phrase_hits) * 4
        )
        candidates.append(
            {
                "idx": idx,
                "rendered": rendered,
                "coverage_terms": coverage_terms,
                "matched_phrases": title_phrase_hits | body_phrase_hits,
                "base_score": base_score,
            }
        )

    if not candidates:
        return []

    selected_indices: set[int] = set()
    selected: list[str] = []
    covered_terms: set[str] = set()
    covered_phrases: set[str] = set()

    # Reserve the best title/body match for the most salient question phrases first.
    for phrase in question_phrases[:HOTPOT_MAX_PASSAGES]:
        best = None
        best_score = -1.0
        for cand in candidates:
            idx = int(cand["idx"])
            if idx in selected_indices:
                continue
            phrase_score = _phrase_match_score(cand["rendered"], phrase)
            if phrase_score > best_score:
                best = cand
                best_score = phrase_score
        if best is not None and best_score > 0.0:
            selected_indices.add(int(best["idx"]))
            selected.append(str(best["rendered"]).strip())
            covered_terms |= set(best["coverage_terms"])
            covered_phrases |= set(best["matched_phrases"])
        if len(selected) >= HOTPOT_MAX_PASSAGES:
            break

    remaining = [cand for cand in candidates if int(cand["idx"]) not in selected_indices]
    while remaining and len(selected) < HOTPOT_MAX_PASSAGES:
        remaining.sort(
            key=lambda row: (
                -(
                    float(row["base_score"])
                    + 10.0 * len(set(row["matched_phrases"]) - covered_phrases)
                    + 4.0 * len(set(row["coverage_terms"]) - covered_terms)
                ),
                int(row["idx"]),
            )
        )
        best = remaining.pop(0)
        selected.append(str(best["rendered"]).strip())
        covered_terms |= set(best["coverage_terms"])
        covered_phrases |= set(best["matched_phrases"])
    return selected


def _compress_hotpot_context(context, question: str, max_chars: int = HOTPOT_MAX_EVIDENCE_CHARS) -> str:
    passages = _ranked_hotpot_passages(context, question)
    if not passages:
        flat = _hotpot_context_to_text(context)
        return flat[:max_chars].strip()
    kept: list[str] = []
    used = 0
    for passage in passages[:HOTPOT_MAX_PASSAGES]:
        text = passage.strip()
        if not text:
            continue
        if used + len(text) + (1 if kept else 0) > max_chars:
            remain = max_chars - used - (1 if kept else 0)
            if remain <= 0:
                break
            text = text[:remain].rstrip()
        kept.append(text)
        used += len(text) + (1 if kept else 0)
        if used >= max_chars:
            break
    return "\n".join(kept).strip()


def build_hotpotqa_prompt(context: str, question: str) -> str:
    return HOTPOT_PROMPT_TEMPLATE.format(context=context.strip(), question=question.strip())


def postprocess_hotpotqa_prediction(text: str) -> str:
    out = str(text or "").strip()
    for prefix in [
        "Answer:",
        "answer:",
        "Question:",
        "question:",
        "Evidence:",
        "evidence:",
    ]:
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
    for sep in ["\n", "</s>", "<|endoftext|>", "Question:", "question:", "Evidence:", "evidence:"]:
        if sep in out:
            out = out.split(sep, 1)[0].strip()
    lowered = out.lower().strip()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return out.strip()


def load_hotpotqa_samples(data_root: str | Path, limit: int | None = None) -> list[HotpotQASample]:
    root = Path(data_root)
    candidates = [
        root / "HotpotQA" / "raw" / "hotpot_dev_distractor_v1.json",
        root / "droidspeak" / "hotpotqa.json",
    ]
    saw_existing_candidate = False
    for path in candidates:
        if not path.exists():
            continue
        saw_existing_candidate = True
        obj = _load_structured_file(path)
        if not isinstance(obj, list):
            continue
        samples: list[HotpotQASample] = []
        for idx, row in enumerate(obj):
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue
            context = _compress_hotpot_context(row.get("context"), question) or "Please answer by multi-hop reasoning from common sense."
            qid = str(row.get("_id") or row.get("id") or row.get("qid") or f"hotpot_{idx}")
            samples.append(
                HotpotQASample(
                    qid=qid,
                    context=context,
                    question=question,
                    answers=[answer],
                    prompt=build_hotpotqa_prompt(context, question),
                )
            )
            if limit is not None and len(samples) >= int(limit):
                return samples
        if samples:
            return samples
    if saw_existing_candidate:
        raise ValueError(f"found HotpotQA file but could not parse a usable list: {candidates}")
    raise FileNotFoundError(f"missing usable HotpotQA dev data: {candidates}")


def _normalize_hotpot_answer(text: str) -> str:
    lower = str(text or "").lower()
    no_punc = "".join(ch for ch in lower if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def hotpotqa_em(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_hotpot_answer(prediction) == _normalize_hotpot_answer(ground_truth) else 0.0


def hotpotqa_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_hotpot_answer(prediction).split()
    gt_tokens = _normalize_hotpot_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gt_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_hotpotqa_prediction(prediction: str, ground_truths: Iterable[str]) -> dict[str, float | str]:
    prediction = postprocess_hotpotqa_prediction(prediction)
    refs = [str(x).strip() for x in ground_truths if str(x).strip()]
    if not refs:
        return {"f1": 0.0, "em": 0.0, "best_answer": ""}
    best_f1 = 0.0
    best_em = 0.0
    best_answer = refs[0]
    for answer in refs:
        f1 = hotpotqa_f1(prediction, answer)
        em = hotpotqa_em(prediction, answer)
        if (f1, em) > (best_f1, best_em):
            best_f1 = f1
            best_em = em
            best_answer = answer
    return {"f1": float(best_f1), "em": float(best_em), "best_answer": best_answer}
