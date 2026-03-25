"""
Tests for the prompt_builder utility.
Run with: pytest tests/test_prompt_builder.py -v
"""
import pytest
from utils.prompt_builder import build_conditioned_prompt, _condense_lyrics


class TestCondenseLyrics:
    def test_short_lyrics_unchanged(self):
        text = "Hello world"
        result = _condense_lyrics(text, max_chars=300)
        assert "Hello world" in result

    def test_multiline_joined_with_slash(self):
        text = "Line one\nLine two\nLine three"
        result = _condense_lyrics(text)
        assert " / " in result

    def test_long_lyrics_truncated(self):
        long_text = "x " * 500
        result = _condense_lyrics(long_text, max_chars=100)
        assert len(result) <= 110  # small slack for ellipsis

    def test_max_four_lines(self):
        text = "\n".join(f"Line {i}" for i in range(10))
        result = _condense_lyrics(text)
        # Should contain at most 4 lines joined by " / "
        assert result.count(" / ") <= 3

    def test_empty_input(self):
        assert _condense_lyrics("") == ""
        assert _condense_lyrics("   ") == ""


class TestBuildConditionedPrompt:
    def test_text_only(self):
        prompt = build_conditioned_prompt("upbeat jazz piano")
        assert "upbeat jazz piano" in prompt

    def test_with_user_lyrics(self):
        prompt = build_conditioned_prompt("rock anthem", user_lyrics="We will rock you")
        assert "We will rock you" in prompt
        assert "rock anthem" in prompt

    def test_user_lyrics_takes_precedence_over_transcribed(self):
        prompt = build_conditioned_prompt(
            "pop song",
            user_lyrics="user lyrics here",
            transcribed_lyrics="transcribed text",
        )
        assert "user lyrics here" in prompt
        assert "transcribed text" not in prompt

    def test_transcribed_lyrics_used_when_no_user_lyrics(self):
        prompt = build_conditioned_prompt(
            "ballad",
            transcribed_lyrics="I will always love you",
        )
        assert "I will always love you" in prompt

    def test_prompt_not_too_long(self):
        long_text = "great music " * 100
        long_lyrics = "la la la " * 200
        prompt = build_conditioned_prompt(long_text, user_lyrics=long_lyrics)
        assert len(prompt) <= 950

    def test_empty_lyrics_ignored(self):
        prompt = build_conditioned_prompt("jazz", user_lyrics="   ", transcribed_lyrics="")
        assert "with lyrics" not in prompt

    def test_empty_prompt_with_lyrics(self):
        prompt = build_conditioned_prompt("", user_lyrics="some words")
        assert "some words" in prompt
