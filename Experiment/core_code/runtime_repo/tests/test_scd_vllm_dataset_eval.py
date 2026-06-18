from __future__ import annotations

from scd_vllm.dataset_eval import _format_prompt


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
