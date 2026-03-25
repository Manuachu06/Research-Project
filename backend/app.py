"""
Music Generation Application - FastAPI Backend

Supports:
  - Text-to-music generation via MusicGen (AudioCraft)
  - Melody-conditioned generation (upload a reference song)
  - Lyrics-aware generation (lyrics embedded in text prompt)
  - Voice input: Whisper transcribes lyrics, Demucs separates melody
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.music_generator import MusicGenerator
from models.audio_processor import AudioProcessor
from utils.prompt_builder import build_conditioned_prompt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Music Generator",
    description=(
        "Generate music from text, voice melody, and lyrics "
        "using MusicGen + Whisper + Demucs"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Output directory for generated files
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_music_generator: Optional[MusicGenerator] = None
_audio_processor: Optional[AudioProcessor] = None


def get_music_generator() -> MusicGenerator:
    global _music_generator
    if _music_generator is None:
        logger.info("Loading MusicGenerator …")
        _music_generator = MusicGenerator()
    return _music_generator


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        logger.info("Loading AudioProcessor …")
        _audio_processor = AudioProcessor()
    return _audio_processor


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    try:
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return {"status": "ok", "device": device, "timestamp": time.time()}


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    """Return available MusicGen model variants."""
    return {
        "models": [
            {
                "id": "facebook/musicgen-small",
                "label": "MusicGen Small",
                "description": "300M params – fast, lower quality",
            },
            {
                "id": "facebook/musicgen-medium",
                "label": "MusicGen Medium",
                "description": "1.5B params – balanced quality/speed",
            },
            {
                "id": "facebook/musicgen-large",
                "label": "MusicGen Large",
                "description": "3.3B params – highest quality, slower",
            },
            {
                "id": "facebook/musicgen-melody",
                "label": "MusicGen Melody",
                "description": "Melody-conditioned generation (best for audio uploads)",
            },
            {
                "id": "facebook/musicgen-stereo-small",
                "label": "MusicGen Stereo Small",
                "description": "Stereo output, 300M params",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Core generation endpoint
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate_music(
    text_prompt: str = Form(..., description="Text description of the desired music"),
    lyrics: Optional[str] = Form(None, description="Lyrics / words to embed in the prompt"),
    duration: float = Form(15.0, description="Output duration in seconds (5–60)"),
    model_id: str = Form("facebook/musicgen-melody", description="MusicGen model variant"),
    audio_file: Optional[UploadFile] = File(None, description="Reference song / voice input"),
    guidance_scale: float = Form(3.0, description="Classifier-free guidance scale (1–10)"),
    top_k: int = Form(250, description="Top-k sampling (50–2000)"),
    temperature: float = Form(1.0, description="Sampling temperature (0.5–2.0)"),
):
    """
    Generate music based on:
    - text_prompt: freeform text describing the music
    - lyrics: optional lyrics to embed into the prompt
    - audio_file: optional reference audio for melody conditioning
                  – Whisper extracts lyrics from voice
                  – Demucs separates melody for MusicGen-Melody conditioning
    """
    # --- Input validation ---
    duration = max(5.0, min(duration, 60.0))
    guidance_scale = max(1.0, min(guidance_scale, 10.0))
    top_k = max(50, min(top_k, 2000))
    temperature = max(0.5, min(temperature, 2.0))

    logger.info(
        "Generate request — model=%s, duration=%.1fs, has_audio=%s",
        model_id,
        duration,
        audio_file is not None,
    )

    gen = get_music_generator()
    proc = get_audio_processor()

    # -----------------------------------------------------------------------
    # Process uploaded audio (voice / reference song)
    # -----------------------------------------------------------------------
    melody_waveform: Optional[np.ndarray] = None
    melody_sr: Optional[int] = None
    transcribed_lyrics: Optional[str] = None

    if audio_file is not None:
        audio_bytes = await audio_file.read()
        suffix = Path(audio_file.filename or "upload.wav").suffix or ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # 1. Transcribe lyrics from the uploaded audio using Whisper
            logger.info("Transcribing audio with Whisper …")
            try:
                transcribed_lyrics = await asyncio.to_thread(proc.transcribe, tmp_path)
                logger.info(
                    "Transcription: %s",
                    transcribed_lyrics[:200] if transcribed_lyrics else "<empty>",
                )
            except Exception as exc:
                logger.warning("Audio transcription failed (%s) — proceeding without transcribed lyrics.", exc)
                transcribed_lyrics = None

            # 2. Separate melody using Demucs
            logger.info("Separating melody with Demucs …")
            try:
                melody_waveform, melody_sr = await asyncio.to_thread(
                    proc.separate_melody, tmp_path
                )
            except Exception as exc:
                logger.warning("Melody separation failed (%s) — proceeding without melody conditioning.", exc)
                melody_waveform = None
                melody_sr = None
        finally:
            os.unlink(tmp_path)

    # -----------------------------------------------------------------------
    # Build the final text prompt
    # -----------------------------------------------------------------------
    final_prompt = build_conditioned_prompt(
        text_prompt=text_prompt,
        user_lyrics=lyrics,
        transcribed_lyrics=transcribed_lyrics,
    )
    logger.info("Final prompt: %s", final_prompt[:300])

    # -----------------------------------------------------------------------
    # Run MusicGen inference
    # -----------------------------------------------------------------------
    output_filename = f"music_{uuid.uuid4().hex[:12]}.wav"
    output_path = OUTPUT_DIR / output_filename

    try:
        await asyncio.to_thread(
            gen.generate,
            prompt=final_prompt,
            output_path=str(output_path),
            duration=duration,
            model_id=model_id,
            melody_waveform=melody_waveform,
            melody_sr=melody_sr,
            guidance_scale=guidance_scale,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Music generation failed: {exc}")

    return JSONResponse(
        {
            "status": "success",
            "filename": output_filename,
            "url": f"/outputs/{output_filename}",
            "download_url": f"/api/download/{output_filename}",
            "prompt_used": final_prompt,
            "transcribed_lyrics": transcribed_lyrics,
            "duration": duration,
            "model_id": model_id,
        }
    )


# ---------------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------------
@app.get("/api/download/{filename}")
async def download_music(filename: str):
    """Download a previously generated audio file."""
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_name,
        headers={"Content-Disposition": f"attachment; filename={safe_name}"},
    )


# ---------------------------------------------------------------------------
# List generated files
# ---------------------------------------------------------------------------
@app.get("/api/files")
async def list_files():
    """Return metadata for all generated audio files."""
    files = []
    for fp in sorted(
        OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        stat = fp.stat()
        files.append(
            {
                "filename": fp.name,
                "url": f"/outputs/{fp.name}",
                "download_url": f"/api/download/{fp.name}",
                "size_bytes": stat.st_size,
                "created_at": stat.st_mtime,
            }
        )
    return {"files": files}


# ---------------------------------------------------------------------------
# Delete a generated file
# ---------------------------------------------------------------------------
@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"status": "deleted", "filename": safe_name}


# ---------------------------------------------------------------------------
# Static file serving — MUST come AFTER all API route registrations
# so that the catch-all "/" mount does not intercept API requests.
# ---------------------------------------------------------------------------
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
