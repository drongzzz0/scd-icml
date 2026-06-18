from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cmrc import CMRCSample, load_cmrc2018_samples


@dataclass
class StrictTextSample:
    qid: str
    context: str
    question: str
    answers: list[str]
    prompt: str


HOTPOT_COMPRESSED_PROMPT_TEMPLATE = (
    "Read the question and answer using only the evidence.\n"
    "Output only the shortest answer.\n"
    "Do not repeat the question or evidence.\n"
    "If the question is yes/no, answer only yes or no.\n"
    "If the answer is a person, place, organization, number, or date, output only that short span.\n"
    "Question: {question}\n"
    "Evidence:\n{context}\n"
    "Answer:"
)
HOTPOT_COMPRESSED_MAX_EVIDENCE_CHARS = 900
HOTPOT_COMPRESSED_MAX_PASSAGES = 4
HOTPOT_COMPRESSED_MAX_SENTENCES_PER_PASSAGE = 3

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
_QUESTION_PHRASE_CONNECTORS = {
    "of",
    "the",
    "and",
    "for",
    "in",
    "on",
    "at",
    "de",
    "la",
    "le",
    "du",
    "van",
    "von",
    "v.",
    "vs.",
}


def _load_json_or_jsonl(path: Path):
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


def _candidate_roots(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    roots = [root]
    if (root / "data").exists():
        roots.append(root / "data")
    return roots


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _hotpot_context_blocks(context) -> list[tuple[str, list[str]]]:
    if isinstance(context, list):
        blocks: list[tuple[str, list[str]]] = []
        for block in context:
            if isinstance(block, list) and len(block) >= 2:
                title = str(block[0])
                sentences = block[1]
                if isinstance(sentences, list):
                    blocks.append((title, [str(x) for x in sentences if str(x).strip()]))
                else:
                    blocks.append((title, [str(sentences)]))
            else:
                text = str(block).strip()
                if text:
                    blocks.append(("", [text]))
        return blocks
    return []


def _hotpot_context_to_text(context) -> str:
    blocks = _hotpot_context_blocks(context)
    if blocks:
        parts: list[str] = []
        for title, sentences in blocks:
            prefix = f"[{title}] " if title else ""
            parts.append(prefix + " ".join(sentences))
        return "\n".join(part for part in parts if part.strip())
    if isinstance(context, dict):
        return json.dumps(context, ensure_ascii=False)
    return str(context or "").strip()


def _hotpotqa_candidate_paths(data_root: str | Path) -> list[Path]:
    roots = _candidate_roots(data_root)
    direct = []
    for root in roots:
        direct.extend(
            [
                root / "HotpotQA" / "raw" / "hotpot_dev_distractor_v1.json",
                root / "hotpotqa" / "hotpot_dev_distractor_v1.json",
                root / "hotpotqa_official" / "hotpot_dev_distractor_v1.json",
                root / "hotpotqa_official" / "hotpot_dev_distractor_v1.full.json",
                root / "tmp" / "hotpotqa_official" / "hotpot_dev_distractor_v1.json",
                root / "tmp" / "hotpotqa_official" / "hotpot_dev_distractor_v1.full.json",
                root / "droidspeak" / "hotpotqa.json",
            ]
        )
    discovered = []
    for root in roots:
        if root.exists():
            discovered.extend(root.rglob("hotpot_dev_distractor_v1*.json"))
    return _unique_paths([*direct, *discovered])


def build_hotpotqa_prompt(context, question: str, supporting_facts=None) -> tuple[str, str]:
    del supporting_facts
    evidence = _hotpot_context_to_text(context).strip()
    if not evidence:
        evidence = "Use the provided facts and answer the question."
    question = str(question or "").strip()
    prompt = (
        "Answer the HotpotQA question using only the evidence.\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}\n"
        f"Question: {question}\n"
        "Give only the final short answer. Use yes or no when appropriate.\n"
        "Answer:"
    )
    return prompt, evidence


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
    return phrases[: max(2, HOTPOT_COMPRESSED_MAX_PASSAGES)]


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
        chosen = sentence_rows[: max(1, HOTPOT_COMPRESSED_MAX_SENTENCES_PER_PASSAGE)]
        chosen.sort(key=lambda row: row[1])
        chosen_sentences = [row[2] for row in chosen if row[2].strip()]
        if not chosen_sentences and sentences:
            chosen_sentences = sentences[:HOTPOT_COMPRESSED_MAX_SENTENCES_PER_PASSAGE]

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

    for phrase in question_phrases[:HOTPOT_COMPRESSED_MAX_PASSAGES]:
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
        if len(selected) >= HOTPOT_COMPRESSED_MAX_PASSAGES:
            break

    remaining = [cand for cand in candidates if int(cand["idx"]) not in selected_indices]
    while remaining and len(selected) < HOTPOT_COMPRESSED_MAX_PASSAGES:
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


def _compress_hotpot_context(context, question: str, max_chars: int = HOTPOT_COMPRESSED_MAX_EVIDENCE_CHARS) -> str:
    passages = _ranked_hotpot_passages(context, question)
    if not passages:
        flat = _hotpot_context_to_text(context)
        return flat[:max_chars].strip()
    kept: list[str] = []
    used = 0
    for passage in passages[:HOTPOT_COMPRESSED_MAX_PASSAGES]:
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


def build_hotpotqa_compressed_prompt(context: str, question: str) -> str:
    return HOTPOT_COMPRESSED_PROMPT_TEMPLATE.format(context=context.strip(), question=question.strip())


def _load_hotpotqa_candidate(path: Path):
    try:
        return _load_json_or_jsonl(path)
    except (json.JSONDecodeError, ValueError):
        return None


def _hotpotqa_question(row: dict) -> str:
    return str(row.get("question") or row.get("input") or "").strip()


def _hotpotqa_answers(row: dict) -> list[str]:
    raw = row.get("answer")
    if raw is None:
        raw = row.get("answers")
    if isinstance(raw, list):
        answers = [str(item).strip() for item in raw if str(item).strip()]
    else:
        answer = str(raw or "").strip()
        answers = [answer] if answer else []
    return answers


def load_hotpotqa_samples(data_root: str | Path, limit: int | None = None) -> list[StrictTextSample]:
    for path in _hotpotqa_candidate_paths(data_root):
        if not path.exists():
            continue
        obj = _load_hotpotqa_candidate(path)
        if not isinstance(obj, list):
            continue
        samples: list[StrictTextSample] = []
        for idx, row in enumerate(obj):
            if not isinstance(row, dict):
                continue
            question = _hotpotqa_question(row)
            answers = _hotpotqa_answers(row)
            if not question or not answers:
                continue
            qid = str(row.get("_id") or row.get("id") or f"hotpotqa_{idx}")
            prompt, context = build_hotpotqa_prompt(row.get("context"), question, row.get("supporting_facts"))
            samples.append(StrictTextSample(qid=qid, context=context, question=question, answers=answers, prompt=prompt))
            if limit is not None and len(samples) >= limit:
                return samples
        if samples:
            return samples
    raise FileNotFoundError(f"no HotpotQA dev file found under {data_root}")


def load_hotpotqa_compressed_samples(data_root: str | Path, limit: int | None = None) -> list[StrictTextSample]:
    for path in _hotpotqa_candidate_paths(data_root):
        if not path.exists():
            continue
        obj = _load_hotpotqa_candidate(path)
        if not isinstance(obj, list):
            continue
        samples: list[StrictTextSample] = []
        for idx, row in enumerate(obj):
            if not isinstance(row, dict):
                continue
            question = _hotpotqa_question(row)
            answers = _hotpotqa_answers(row)
            if not question or not answers:
                continue
            qid = str(row.get("_id") or row.get("id") or row.get("qid") or f"hotpotqa_{idx}")
            context = _compress_hotpot_context(row.get("context"), question)
            if not context:
                context = "Please answer by multi-hop reasoning from common sense."
            samples.append(
                StrictTextSample(
                    qid=qid,
                    context=context,
                    question=question,
                    answers=answers,
                    prompt=build_hotpotqa_compressed_prompt(context, question),
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
        if samples:
            return samples
    raise FileNotFoundError(f"no HotpotQA dev file found under {data_root}")


def load_wikitext2_samples(
    data_root: str | Path,
    limit: int | None = None,
    *,
    prefix_words: int = 96,
    target_words: int = 32,
) -> list[StrictTextSample]:
    roots = _candidate_roots(data_root)
    direct = []
    for root in roots:
        direct.extend(
            [
                root / "wikitext-2" / "wiki.test.txt",
                root / "WikiText-2" / "wiki.test.txt",
                root / "wikitext" / "wiki.test.txt",
            ]
        )
    discovered = []
    for root in roots:
        if root.exists():
            discovered.extend(root.rglob("wiki.test.txt"))
    paths = _unique_paths([*direct, *discovered])

    for path in paths:
        if not path.exists():
            continue
        samples: list[StrictTextSample] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            text = raw.strip()
            if not text or text.startswith("="):
                continue
            words = text.split()
            need = int(prefix_words) + int(target_words)
            if len(words) < need:
                continue
            prefix = " ".join(words[: int(prefix_words)])
            target = " ".join(words[int(prefix_words) : need])
            qid = f"wikitext2_{path.name}_{line_no}"
            samples.append(
                StrictTextSample(
                    qid=qid,
                    context=prefix,
                    question="continue",
                    answers=[target],
                    prompt=prefix,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
        if samples:
            return samples
    raise FileNotFoundError(f"no WikiText-2 wiki.test.txt found under {data_root}")


def load_text_benchmark_samples(
    dataset: str,
    data_root: str | Path,
    limit: int | None = None,
    *,
    cmrc_split: str = "dev",
):
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "cmrc2018":
        for root in _candidate_roots(data_root):
            try:
                return load_cmrc2018_samples(root, limit=limit, split=cmrc_split)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"no CMRC2018 {cmrc_split} file found under {data_root}")
    if dataset_key == "hotpotqa":
        return load_hotpotqa_samples(data_root, limit=limit)
    if dataset_key == "hotpotqa_compressed":
        return load_hotpotqa_compressed_samples(data_root, limit=limit)
    if dataset_key in {"wikitext2", "wikitext-2"}:
        return load_wikitext2_samples(data_root, limit=limit)
    raise ValueError(f"unsupported dataset: {dataset}")


__all__ = [
    "CMRCSample",
    "StrictTextSample",
    "build_hotpotqa_compressed_prompt",
    "build_hotpotqa_prompt",
    "load_hotpotqa_compressed_samples",
    "load_hotpotqa_samples",
    "load_text_benchmark_samples",
    "load_wikitext2_samples",
]
