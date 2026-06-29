from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable


CMRC_QWEN_SHORT_SYSTEM = (
    "你是中文阅读理解助手。回答时只输出材料中的最短答案短语，"
    "不要解释，不要完整句子。"
)


def build_cmrc_short_raw_prompt(context: str, question: str) -> str:
    return (
        "请阅读材料并回答问题。只从材料中抽取答案，"
        "只输出最短答案短语，不要解释，不要重复问题，不要输出完整句子。\n"
        f"材料：{str(context).strip()}\n"
        f"问题：{str(question).strip()}\n"
        "答案："
    )


def build_cmrc_qwen_chat_short_prompt(tokenizer: Any, context: str, question: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("cmrc_qwen_chat_short requires a tokenizer with apply_chat_template")
    messages = [
        {"role": "system", "content": CMRC_QWEN_SHORT_SYSTEM},
        {"role": "user", "content": f"材料：{str(context).strip()}\n问题：{str(question).strip()}"},
    ]
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def normalize_prompt_style(style: str | None) -> str:
    return str(style or "").strip().lower().replace("-", "_")


def apply_prompt_style_to_samples(
    samples: Iterable[Any],
    *,
    style: str | None,
    tokenizer: Any | None = None,
    dataset: str = "",
) -> list[Any]:
    normalized = normalize_prompt_style(style)
    rows = list(samples)
    if normalized in {"", "default", "cmrc_default"}:
        return rows

    dataset_key = str(dataset or "").strip().lower()
    if dataset_key and dataset_key not in {"cmrc2018", "cmrc"}:
        raise ValueError(f"prompt_style={style!r} is only supported for CMRC2018, got dataset={dataset!r}")

    out: list[Any] = []
    for sample in rows:
        context = getattr(sample, "context")
        question = getattr(sample, "question")
        if normalized == "cmrc_short_raw":
            prompt = build_cmrc_short_raw_prompt(context, question)
        elif normalized in {"cmrc_qwen_chat_short", "qwen_chat_short"}:
            if tokenizer is None:
                raise ValueError("cmrc_qwen_chat_short requires tokenizer")
            prompt = build_cmrc_qwen_chat_short_prompt(tokenizer, context, question)
        else:
            raise ValueError(f"unsupported prompt_style: {style!r}")
        out.append(replace(sample, prompt=prompt))
    return out
