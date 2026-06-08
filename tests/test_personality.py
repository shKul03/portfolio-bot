import pytest
from app.personality.prompts import PERSONALITY_PROMPTS, VALID_PERSONALITIES

GROUNDING_KEYWORDS = ["kulkarni.shloka03@gmail.com", "context", "invent"]


def test_all_four_keys_exist():
    for key in ("professional", "witty", "hype", "eli5"):
        assert key in PERSONALITY_PROMPTS, f"Missing personality key: {key}"


def test_valid_personalities_set_matches():
    assert VALID_PERSONALITIES == {"professional", "witty", "hype", "eli5"}


def test_each_prompt_is_non_empty_string():
    for key, prompt in PERSONALITY_PROMPTS.items():
        assert isinstance(prompt, str), f"{key} prompt is not a string"
        assert len(prompt.strip()) > 0, f"{key} prompt is empty"


def test_each_prompt_contains_grounding_email():
    for key, prompt in PERSONALITY_PROMPTS.items():
        assert "kulkarni.shloka03@gmail.com" in prompt, (
            f"{key} prompt missing contact email grounding instruction"
        )


def test_each_prompt_contains_no_invent_instruction():
    for key, prompt in PERSONALITY_PROMPTS.items():
        assert "invent" in prompt.lower() or "never" in prompt.lower(), (
            f"{key} prompt missing 'never invent' grounding instruction"
        )


def test_each_prompt_contains_context_instruction():
    for key, prompt in PERSONALITY_PROMPTS.items():
        assert "context" in prompt.lower(), (
            f"{key} prompt missing 'context' instruction"
        )
