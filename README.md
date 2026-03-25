# 🎵 AI Music Generator

An end-to-end music generation application powered by **Meta's AudioCraft MusicGen**, **OpenAI Whisper**, and **Demucs**. Generate original music from text descriptions, uploaded voice / song files, and optional lyrics — then listen and download directly in the browser.

---

## ✨ Features

| Capability | Technology |
|---|---|
| Text-to-music generation | MusicGen (Small / Medium / Large / Melody) |
| Melody-conditioned generation | MusicGen-Melody + Demucs source separation |
| Voice / song upload → melody conditioning | Demucs `htdemucs` model |
| Lyrics extraction from uploaded audio | OpenAI Whisper |
| In-browser audio player with waveform | WaveSurfer.js |
| Download generated WAV files | FastAPI file serving |
| Generated file history & management | FastAPI + filesystem |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser (Frontend)                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  index.html + app.js + style.css                    │ │
│  │  • Text prompt input                                │ │
│  │  • Lyrics textarea                                  │ │
│  │  • Drag-and-drop audio upload                      │ │
│  │  • WaveSurfer.js audio player                      │ │
│  │  • Download & history management                   │ │
│  └─────────────────────┬───────────────────────────────┘ │
└────────────────────────┼────────────────────────────────┘
                         │ HTTP (multipart/form-data)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                        │
│                                                          │
│   POST /api/generate                                     │
│   ┌──────────────────────────────────────────────────┐  │
│   │ 1. Audio upload?                                  │  │
│   │    ├── Whisper → transcribe lyrics               │  │
│   │    └── Demucs  → separate melody stem            │  │
│   │ 2. Build conditioned prompt                       │  │
│   │    (text + lyrics + transcribed lyrics)           │  │
│   │ 3. MusicGen inference                             │  │
│   │    ├── text-only  (all models)                   │  │
│   │    └── melody+text (musicgen-melody)              │  │
│   │ 4. Save WAV → outputs/                            │  │
│   └──────────────────────────────────────────────────┘  │
│                                                          │
│   GET  /outputs/{file}    — stream audio                 │
│   GET  /api/download/{f}  — force-download               │
│   GET  /api/files         — list history                 │
│   DELETE /api/files/{f}   — delete a file                │
└─────────────────────────────────────────────────────────┘
```

### Models Used

| Model | Purpose | Notes |
|---|---|---|
| **MusicGen-Melody** (Meta AudioCraft) | Melody + text conditioned music generation | Primary model — accepts a reference audio |
| **MusicGen-Small / Medium / Large** | Text-to-music generation | Smaller models for faster generation |
| **Demucs `htdemucs`** | Source separation (extract instruments from song) | Separates bass, drums, other from vocals |
| **OpenAI Whisper** | Speech-to-text transcription | Extracts lyrics from voice / song uploads |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `ffmpeg` installed and on `PATH` (required by Whisper and Demucs)
- A CUDA-capable GPU is **strongly recommended** (CPU inference is very slow for large models)

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note on PyTorch**: `audiocraft` requires PyTorch ≥ 2.1. Install the correct CUDA variant from https://pytorch.org/get-started/locally/ **before** running `pip install -r requirements.txt`.

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env to customise host, port, default model, etc.
```

### 3. Start the server

```bash
python main.py
# Or with auto-reload for development:
MUSIC_GEN_RELOAD=true python main.py
```

The server starts at **http://localhost:8000**.  
The frontend is served automatically at the root URL.

---

## 🎛️ Using the Application

### Input Methods

1. **Text Description** *(required)*  
   Describe the music you want:  
   `"Upbeat lo-fi hip-hop with mellow piano, soft drums and a warm bassline"`

2. **Reference Audio / Voice** *(optional)*  
   Upload an MP3, WAV, OGG, or FLAC file. The system will:
   - Transcribe any spoken/sung lyrics using **Whisper**
   - Extract the melody/instrumental stem using **Demucs**
   - Feed the melody into **MusicGen-Melody** for conditioning

3. **Lyrics** *(optional)*  
   Paste in lyrics to influence the style, mood, and phrasing of the generated music. If you also upload an audio file, your typed lyrics take precedence over Whisper's transcription.

### Model Selection

| Model | Best For |
|---|---|
| MusicGen Melody | Audio uploads — melody conditioning |
| MusicGen Small | Quick previews, low VRAM |
| MusicGen Medium | Good balance of quality & speed |
| MusicGen Large | Highest quality, needs ~16GB VRAM |
| MusicGen Stereo Small | Stereo output, fast |

### Advanced Settings

| Setting | Effect |
|---|---|
| Duration | Length of generated audio (5–60 seconds) |
| Guidance Scale | How strictly the model follows the prompt (higher = more faithful) |
| Top-K | Number of tokens considered at each sampling step |
| Temperature | Randomness / creativity (higher = more varied) |

---

## 🐳 Docker

```bash
docker build -t music-generator .
docker run -p 8000:8000 --gpus all music-generator
```

For CPU-only:
```bash
docker run -p 8000:8000 music-generator
```

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest httpx scipy
pytest tests/ -v
```

---

## 📁 Project Structure

```
Research-Project/
├── backend/
│   ├── app.py                  # FastAPI application & endpoints
│   ├── main.py                 # Entry point (uvicorn)
│   ├── config.py               # Settings (env vars)
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/
│   │   ├── music_generator.py  # MusicGen wrapper
│   │   └── audio_processor.py  # Whisper + Demucs wrapper
│   ├── utils/
│   │   └── prompt_builder.py   # Prompt conditioning logic
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_prompt_builder.py
│   └── outputs/                # Generated WAV files (gitignored)
├── frontend/
│   ├── index.html              # Single-page application
│   ├── style.css               # Dark glassmorphism UI
│   └── app.js                  # WaveSurfer player + API calls
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔬 Research Background

This project draws on several research directions:

- **MusicGen** (Copet et al., 2023 — *"Simple and Controllable Music Generation"*): A single-stage autoregressive model conditioned on text and optionally on a melody chroma representation extracted from a reference audio.
- **AudioCraft** (Meta, 2023): Open-source framework housing MusicGen, AudioGen, and EnCodec.
- **Demucs** (Défossez et al., 2021 — *"Hybrid Transformers for Music Source Separation"*): State-of-the-art music source separation using a hybrid time-domain + spectrogram approach.
- **Whisper** (Radford et al., 2022 — *"Robust Speech Recognition via Large-Scale Weak Supervision"*): A large-scale multilingual speech recognition model with strong performance on music/song transcription.

---

## ⚠️ Notes

- First run downloads model weights (~300 MB for `musicgen-small`, ~3 GB for `musicgen-large`). Ensure you have enough disk space.
- Generation time depends heavily on hardware. On a modern GPU, 15 seconds of music takes ~5–15 seconds; on CPU, expect 5–10 minutes.
- The `musicgen-melody` model requires the audio stem from Demucs to be at the model's native 32 kHz sample rate — resampling is handled automatically.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
