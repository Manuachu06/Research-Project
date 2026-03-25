"""
MusicGenerator — wraps Meta's AudioCraft MusicGen models.

Supports:
  - Text-only generation (all MusicGen variants)
  - Melody-conditioned generation (musicgen-melody)

Model loading is cached per model_id to avoid reloading between requests.
All heavy dependencies (torch, torchaudio, audiocraft) are imported lazily
so the module can be imported in environments without GPU packages installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cache: model_id → loaded MusicGen instance
_model_cache: Dict[str, object] = {}

# Model IDs that support melody conditioning
_MELODY_MODELS = {"facebook/musicgen-melody", "facebook/musicgen-stereo-melody"}


def _get_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class MusicGenerator:
    """Thin wrapper around AudioCraft MusicGen for inference."""

    def __init__(self) -> None:
        logger.info("MusicGenerator initialised (device=%s)", _get_device())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        output_path: str,
        duration: float = 15.0,
        model_id: str = "facebook/musicgen-melody",
        melody_waveform: Optional[np.ndarray] = None,
        melody_sr: Optional[int] = None,
        guidance_scale: float = 3.0,
        top_k: int = 250,
        temperature: float = 1.0,
    ) -> None:
        """
        Generate music and save as a WAV file.

        Parameters
        ----------
        prompt          : Text description / conditioned prompt.
        output_path     : Destination WAV file path.
        duration        : Output length in seconds.
        model_id        : HuggingFace model identifier.
        melody_waveform : (Optional) Reference melody as numpy array (C×T or 1D).
        melody_sr       : Sample-rate of melody_waveform.
        guidance_scale  : Classifier-free guidance scale.
        top_k           : Top-k for sampling.
        temperature     : Sampling temperature.
        """
        import torch  # type: ignore
        import torchaudio  # type: ignore

        device = _get_device()
        model = self._load_model(model_id)

        # Configure generation parameters
        model.set_generation_params(
            duration=duration,
            guidance_scale=guidance_scale,
            top_k=top_k,
            temperature=temperature,
        )

        # ------------------------------------------------------------------
        # Melody conditioning
        # ------------------------------------------------------------------
        if melody_waveform is not None and model_id in _MELODY_MODELS:
            audio_tensor = self._numpy_to_tensor(melody_waveform, melody_sr)
            logger.info(
                "Melody conditioning: shape=%s, sr=%s", audio_tensor.shape, melody_sr
            )
            # MusicGen-Melody: generate_with_chroma expects a (B, C, T) tensor
            wav = model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=audio_tensor,
                melody_sample_rate=melody_sr,
            )
        else:
            if melody_waveform is not None:
                logger.warning(
                    "Model %s does not support melody conditioning — using text-only.",
                    model_id,
                )
            wav = model.generate(descriptions=[prompt])

        # wav shape: (B, C, T) — take first batch item
        audio_data = wav[0].cpu()  # (C, T)

        # Save as WAV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            audio_data,
            sample_rate=model.sample_rate,
        )
        logger.info("Saved generated music → %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_id: str):
        """Load (and cache) a MusicGen model."""
        if model_id not in _model_cache:
            try:
                from audiocraft.models import MusicGen  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "audiocraft is not installed. Run: pip install audiocraft"
                ) from exc

            device = _get_device()
            logger.info("Loading model %s to %s …", model_id, device)
            model = MusicGen.get_pretrained(model_id)
            model = model.to(device)
            _model_cache[model_id] = model
            logger.info("Model %s loaded.", model_id)
        return _model_cache[model_id]

    @staticmethod
    def _numpy_to_tensor(
        waveform: np.ndarray, sample_rate: Optional[int]
    ):
        """Convert a numpy waveform to a (1, C, T) float tensor."""
        import torch  # type: ignore

        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # (1, T)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T  # ensure (C, T)
        tensor = torch.from_numpy(waveform).float()  # (C, T)
        return tensor.unsqueeze(0)  # (1, C, T)
