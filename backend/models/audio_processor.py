"""
AudioProcessor — Handles:
  1. Speech-to-text transcription via Whisper
  2. Audio source separation (melody extraction) via Demucs

All heavy dependencies are imported lazily so the module can be loaded
in environments without GPU packages installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy-loaded singletons
# ------------------------------------------------------------------
_whisper_model = None


def _get_whisper(model_size: str = "base"):
    """Load and cache a Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            ) from exc
        logger.info("Loading Whisper model '%s' …", model_size)
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper loaded.")
    return _whisper_model


class AudioProcessor:
    """Handles audio transcription and melody separation."""

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, whisper_model_size: str = "base") -> str:
        """
        Transcribe speech/vocals from an audio file using Whisper.

        Parameters
        ----------
        audio_path : Path to the audio file (any format supported by ffmpeg).
        whisper_model_size : Whisper variant — 'tiny', 'base', 'small', 'medium', 'large'.

        Returns
        -------
        Transcribed text (may be empty string if no speech detected).
        """
        whisper_model = _get_whisper(whisper_model_size)
        logger.info("Transcribing %s …", audio_path)
        result = whisper_model.transcribe(audio_path, fp16=False)
        text = result.get("text", "").strip()
        logger.info("Transcription complete: %d chars", len(text))
        return text

    # ------------------------------------------------------------------
    # Melody separation
    # ------------------------------------------------------------------

    def separate_melody(
        self, audio_path: str
    ) -> Tuple[np.ndarray, int]:
        """
        Separate the instrumental/melody stem from an audio file using Demucs.

        Returns the "no_vocals" (instruments-only) waveform as a numpy array
        along with its sample rate.  Falls back to the raw waveform if Demucs
        fails so that generation can still proceed.

        Returns
        -------
        (waveform, sample_rate)  where waveform is shape (C, T) float32.
        """
        try:
            return self._demucs_separate(audio_path)
        except Exception as exc:
            logger.warning(
                "Demucs separation failed (%s) — using original audio for melody.", exc
            )
            return self._load_audio(audio_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _demucs_separate(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Run Demucs htdemucs model and return the no-vocals stem."""
        try:
            import demucs.api  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "demucs is not installed. Run: pip install demucs"
            ) from exc

        logger.info("Running Demucs source separation on %s …", audio_path)

        separator = demucs.api.Separator(model="htdemucs")
        origin, separated = separator.separate_audio_file(Path(audio_path))

        # separated is a dict of stem_name → Tensor (C, T) on device
        # We want the mixture minus vocals (bass + drums + other)
        stems_to_mix = ["bass", "drums", "other"]
        available_stems = list(separated.keys())
        logger.info("Available Demucs stems: %s", available_stems)

        mix_stems = [s for s in stems_to_mix if s in separated]
        if not mix_stems:
            if "no_vocals" in separated:
                melody_tensor = separated["no_vocals"].cpu()
            else:
                tensors = [v.cpu() for k, v in separated.items() if k != "vocals"]
                if not tensors:
                    raise RuntimeError("No usable stems found")
                melody_tensor = sum(tensors)
        else:
            melody_tensor = sum(separated[s].cpu() for s in mix_stems)

        # Normalize to [-1, 1]
        melody_np = melody_tensor.numpy()  # (C, T)
        max_val = np.abs(melody_np).max()
        if max_val > 0:
            melody_np = melody_np / max_val

        sample_rate = separator.samplerate
        logger.info("Demucs done: shape=%s, sr=%s", melody_np.shape, sample_rate)
        return melody_np, sample_rate

    @staticmethod
    def _load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio as float32 numpy array (C, T), returning (array, sr)."""
        try:
            import torchaudio  # type: ignore

            waveform, sr = torchaudio.load(audio_path)
            return waveform.numpy(), sr
        except Exception:
            import soundfile as sf  # type: ignore

            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            return data.T, sr  # (C, T)
