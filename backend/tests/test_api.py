"""
Tests for the FastAPI application endpoints.
Uses TestClient (no real model inference required).
Run with: pytest tests/test_api.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from fastapi.testclient import TestClient

# ============================================================
# Helpers
# ============================================================

def _make_dummy_wav(path: str, sample_rate: int = 32000, duration: float = 0.5):
    """Write a tiny silent WAV to *path*."""
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    wavfile.write(path, sample_rate, samples)


def _fake_generate(self, prompt, output_path, **kwargs):
    """Fake MusicGenerator.generate — writes a silent WAV."""
    _make_dummy_wav(output_path)


def _fake_transcribe(self, audio_path, **kwargs):
    return "fake transcription"


def _fake_separate(self, audio_path):
    return np.zeros((1, 16000), dtype=np.float32), 16000


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def app_with_mocks():
    """
    Import the app with all heavy ML methods patched.
    Uses module-level scope so the app is only imported once.
    """
    with (
        patch("models.music_generator.MusicGenerator.generate", _fake_generate),
        patch("models.audio_processor.AudioProcessor.transcribe", _fake_transcribe),
        patch("models.audio_processor.AudioProcessor.separate_melody", _fake_separate),
    ):
        import app as app_module
        yield app_module


@pytest.fixture
def client(app_with_mocks, tmp_path, monkeypatch):
    """TestClient backed by a temp output directory."""
    monkeypatch.setattr(app_with_mocks, "OUTPUT_DIR", tmp_path)
    return TestClient(app_with_mocks.app)


# ============================================================
# Tests
# ============================================================

class TestHealthEndpoint:
    def test_health_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "device" in data


class TestModelsEndpoint:
    def test_lists_models(self, client):
        res = client.get("/api/models")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        assert len(data["models"]) > 0
        assert all("id" in m and "label" in m for m in data["models"])


class TestGenerateEndpoint:
    def test_generate_text_only(self, client):
        res = client.post(
            "/api/generate",
            data={"text_prompt": "calm jazz piano"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["filename"].endswith(".wav")

    def test_generate_with_lyrics(self, client):
        res = client.post(
            "/api/generate",
            data={
                "text_prompt": "upbeat pop",
                "lyrics": "I love the sunshine every day",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "I love the sunshine" in data["prompt_used"]

    def test_generate_with_audio(self, client, tmp_path):
        wav_path = tmp_path / "sample.wav"
        _make_dummy_wav(str(wav_path))

        with open(wav_path, "rb") as f:
            res = client.post(
                "/api/generate",
                data={"text_prompt": "rock ballad"},
                files={"audio_file": ("sample.wav", f, "audio/wav")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["transcribed_lyrics"] == "fake transcription"

    def test_generate_duration_clamped(self, client):
        res = client.post(
            "/api/generate",
            data={"text_prompt": "test", "duration": "200"},
        )
        assert res.status_code == 200
        assert res.json()["duration"] == 60.0

    def test_generate_missing_prompt_fails(self, client):
        res = client.post("/api/generate", data={})
        assert res.status_code == 422


class TestFilesEndpoint:
    def test_list_files_empty(self, client):
        res = client.get("/api/files")
        assert res.status_code == 200
        assert res.json()["files"] == []

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/files/doesnotexist.wav")
        assert res.status_code == 404

    def test_download_nonexistent(self, client):
        res = client.get("/api/download/nosuchfile.wav")
        assert res.status_code == 404

    def test_list_files_after_generate(self, client):
        client.post("/api/generate", data={"text_prompt": "test music"})
        res = client.get("/api/files")
        assert res.status_code == 200
        files = res.json()["files"]
        assert len(files) >= 1
        assert files[0]["filename"].endswith(".wav")

    def test_delete_file(self, client):
        gen_res = client.post("/api/generate", data={"text_prompt": "test"})
        filename = gen_res.json()["filename"]

        del_res = client.delete(f"/api/files/{filename}")
        assert del_res.status_code == 200

        list_res = client.get("/api/files")
        filenames = [f["filename"] for f in list_res.json()["files"]]
        assert filename not in filenames

    def test_path_traversal_blocked(self, client):
        res = client.get("/api/download/../../etc/passwd")
        assert res.status_code in (404, 400)
