"""
prompt_builder — Builds a rich, conditioned text prompt for MusicGen.

Combines:
  - user-supplied text description
  - user-supplied lyrics (optional)
  - transcribed lyrics from a voice/song upload (optional)
"""

from __future__ import annotations

import re
from typing import Optional


def build_conditioned_prompt(
    text_prompt: str,
    user_lyrics: Optional[str] = None,
    transcribed_lyrics: Optional[str] = None,
) -> str:
    """
    Merge all available signals into a single high-quality MusicGen prompt.

    MusicGen responds well to structured prompts that describe:
      - Musical style / genre
      - Mood / emotion
      - Instrumentation
      - Tempo / energy
      - Lyrics / vocal content

    Returns
    -------
    A single string prompt ready for model.generate().
    """
    parts: list[str] = []

    # 1. Core text description (always present)
    clean_text = text_prompt.strip()
    if clean_text:
        parts.append(clean_text)

    # 2. User-supplied lyrics take precedence over transcribed ones
    lyrics_text: Optional[str] = None
    if user_lyrics and user_lyrics.strip():
        lyrics_text = user_lyrics.strip()
    elif transcribed_lyrics and transcribed_lyrics.strip():
        lyrics_text = transcribed_lyrics.strip()

    if lyrics_text:
        # Condense the lyrics to a short representative excerpt to keep the
        # prompt from overflowing the model's context window.
        excerpt = _condense_lyrics(lyrics_text, max_chars=300)
        parts.append(f"with lyrics: {excerpt}")

    prompt = ", ".join(parts)

    # Ensure the prompt is not too long for the model tokenizer (≈ 1024 chars)
    if len(prompt) > 900:
        prompt = prompt[:900].rsplit(",", 1)[0]

    return prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _condense_lyrics(lyrics: str, max_chars: int = 300) -> str:
    """
    Return a condensed version of the lyrics suitable for embedding in a prompt.

    Strategy:
      1. Strip excessive whitespace.
      2. Take the first verse / chorus (first ~4 lines).
      3. Truncate to max_chars.
    """
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", lyrics).strip()
    # Split into non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Take first 4 lines as representative excerpt
    excerpt = " / ".join(lines[:4])
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0] + "…"
    return excerpt
