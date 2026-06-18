from __future__ import annotations

from types import SimpleNamespace

from scd_vllm.dataset_eval import _build_nll_scoring_prompt, _format_prompt, _lookup_prompt_logprob, parse_args


class FakeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages) + "\nassistant:"


def test_format_prompt_plain_with_extra_instruction() -> None:
    assert _format_prompt(
        "Question:",
        chat_tokenizer=None,
        extra_instruction="Only answer the span.",
    ) == "Question:\nOnly answer the span."


def test_format_prompt_uses_chat_template_with_system_message() -> None:
    rendered = _format_prompt(
        "Question:",
        chat_tokenizer=FakeChatTokenizer(),
        system_prompt="You are concise.",
        extra_instruction="Only answer the span.",
    )

    assert "You are concise." in str(rendered)
    assert "Question:\nOnly answer the span." in str(rendered)
    assert str(rendered).endswith("assistant:")


def test_parse_args_accepts_non_cmrc_datasets() -> None:
    args = parse_args(
        [
            "--dataset",
            "wikitext2",
            "--data-root",
            "/data",
            "--model",
            "/model",
            "--out",
            "/out",
        ]
    )

    assert args.dataset == "wikitext2"


def test_build_nll_scoring_prompt_tracks_target_tokens() -> None:
    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens: bool):
            assert add_special_tokens is True
            return {"input_ids": [0, *range(1, len(str(text).split()) + 1)]}

    sample = SimpleNamespace(qid="w1", prompt="alpha beta", answers=["gamma delta"])
    row = _build_nll_scoring_prompt(sample, FakeTokenizer())

    assert row["text"] == "alpha beta gamma delta"
    assert row["target_start"] == 3
    assert row["target_stop"] == 5
    assert row["target_token_count"] == 2


def test_lookup_prompt_logprob_handles_vllm_objects_and_dicts() -> None:
    assert _lookup_prompt_logprob({7: SimpleNamespace(logprob=-1.25)}, 7) == -1.25
    assert _lookup_prompt_logprob({"8": {"logprob": -2.5}}, 8) == -2.5
    assert _lookup_prompt_logprob({9: -3.0}, 9) == -3.0
    assert _lookup_prompt_logprob({}, 1) is None
