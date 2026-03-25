/**
 * AI Music Generator — Frontend JavaScript
 *
 * Handles:
 *   - Form submission → POST /api/generate
 *   - WaveSurfer audio player
 *   - Drag-and-drop / browse file upload
 *   - Generated file history
 *   - Toast notifications
 */

"use strict";

// ============================================================
// Config
// ============================================================
const API_BASE = "";   // Same origin — served by FastAPI

// ============================================================
// DOM refs
// ============================================================
const healthBadge   = document.getElementById("health-badge");
const healthText    = document.getElementById("health-text");
const form          = document.getElementById("generate-form");
const generateBtn   = document.getElementById("generate-btn");
const btnLabel      = document.getElementById("btn-label");
const btnSpinner    = document.getElementById("btn-spinner");
const progressSec   = document.getElementById("progress-section");
const progressBar   = document.getElementById("progress-bar");
const progressText  = document.getElementById("progress-text");
const resultCard    = document.getElementById("result-card");
const inputCard     = document.getElementById("input-card");
const resultMeta    = document.getElementById("result-meta");
const waveformDiv   = document.getElementById("waveform");
const playPauseBtn  = document.getElementById("play-pause-btn");
const playIcon      = document.getElementById("play-icon");
const currentTimeEl = document.getElementById("current-time");
const totalTimeEl   = document.getElementById("total-time");
const volumeSlider  = document.getElementById("volume-slider");
const downloadLink  = document.getElementById("download-link");
const generateAnother = document.getElementById("generate-another");
const historyList   = document.getElementById("history-list");
const refreshBtn    = document.getElementById("refresh-history");
const toast         = document.getElementById("toast");
const dropZone      = document.getElementById("drop-zone");
const audioUpload   = document.getElementById("audio-upload");
const browseBtn     = document.getElementById("browse-btn");
const filePreview   = document.getElementById("file-preview");
const fileNameDisplay = document.getElementById("file-name-display");
const removeAudioBtn  = document.getElementById("remove-audio");

// Range inputs
const durationSlider  = document.getElementById("duration");
const guidanceSlider  = document.getElementById("guidance-scale");
const topkSlider      = document.getElementById("top-k");
const tempSlider      = document.getElementById("temperature");
const durationVal     = document.getElementById("duration-val");
const guidanceVal     = document.getElementById("guidance-val");
const topkVal         = document.getElementById("topk-val");
const tempVal         = document.getElementById("temp-val");

// ============================================================
// WaveSurfer instance
// ============================================================
let wavesurfer = null;

function initWaveSurfer(audioUrl) {
  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }

  wavesurfer = WaveSurfer.create({
    container: waveformDiv,
    waveColor: "#7c3aed",
    progressColor: "#a78bfa",
    cursorColor: "#fff",
    barWidth: 2,
    barRadius: 2,
    barGap: 1,
    height: 80,
    url: audioUrl,
    backend: "WebAudio",
    interact: true,
  });

  wavesurfer.on("ready", () => {
    totalTimeEl.textContent = formatTime(wavesurfer.getDuration());
    volumeSlider.value = 0.8;
    wavesurfer.setVolume(0.8);
  });

  wavesurfer.on("timeupdate", (t) => {
    currentTimeEl.textContent = formatTime(t);
  });

  wavesurfer.on("finish", () => {
    playIcon.textContent = "▶";
  });

  wavesurfer.on("play", ()  => { playIcon.textContent = "⏸"; });
  wavesurfer.on("pause", () => { playIcon.textContent = "▶"; });
}

// ============================================================
// Health check
// ============================================================
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    healthBadge.className = "health-badge ok";
    healthText.textContent = `Backend online · ${data.device.toUpperCase()}`;
  } catch {
    healthBadge.className = "health-badge error";
    healthText.textContent = "Backend unreachable";
  }
}

// ============================================================
// Range slider labels
// ============================================================
function bindRangeLabel(slider, label, suffix = "") {
  slider.addEventListener("input", () => {
    label.textContent = slider.value + suffix;
  });
}
bindRangeLabel(durationSlider, durationVal, "s");
bindRangeLabel(guidanceSlider, guidanceVal);
bindRangeLabel(topkSlider,     topkVal);
bindRangeLabel(tempSlider,     tempVal);

// ============================================================
// File upload / drag-and-drop
// ============================================================
browseBtn.addEventListener("click", () => audioUpload.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== removeAudioBtn && e.target !== browseBtn) audioUpload.click();
});

audioUpload.addEventListener("change", () => {
  if (audioUpload.files.length) showFilePreview(audioUpload.files[0].name);
});

removeAudioBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  audioUpload.value = "";
  filePreview.classList.add("hidden");
  dropZone.querySelector(".drop-zone-inner").classList.remove("hidden");
});

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave",     () => { dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("audio/")) {
    // Assign to the file input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    audioUpload.files = dataTransfer.files;
    showFilePreview(file.name);
  } else {
    showToast("Please drop an audio file.", "error");
  }
});

function showFilePreview(name) {
  fileNameDisplay.textContent = name;
  filePreview.classList.remove("hidden");
  dropZone.querySelector(".drop-zone-inner").classList.add("hidden");
}

// ============================================================
// Form submission
// ============================================================
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await generateMusic();
});

async function generateMusic() {
  setGenerating(true);

  const formData = new FormData(form);

  // Add sliders that might not be included automatically
  // (FormData only includes named elements)
  const progressMessages = [
    "Preparing models…",
    "Transcribing audio with Whisper…",
    "Separating melody with Demucs…",
    "Running MusicGen inference…",
    "Finalizing audio output…",
  ];
  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    if (msgIdx < progressMessages.length - 1) msgIdx++;
    progressText.textContent = progressMessages[msgIdx];
  }, 8000);

  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      body: formData,
    });

    clearInterval(msgInterval);

    if (!res.ok) {
      let detail = "Generation failed.";
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }

    const data = await res.json();
    showResult(data);
    showToast("Music generated successfully! 🎵", "success");
    loadHistory();

  } catch (err) {
    clearInterval(msgInterval);
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
  } finally {
    setGenerating(false);
  }
}

// ============================================================
// Show result
// ============================================================
function showResult(data) {
  // Metadata
  let metaHtml = `<strong>Prompt used:</strong> ${escHtml(data.prompt_used)}`;
  if (data.transcribed_lyrics) {
    metaHtml += `<br><strong>Transcribed lyrics:</strong> ${escHtml(data.transcribed_lyrics.slice(0, 300))}`;
  }
  metaHtml += `<br><strong>Model:</strong> ${escHtml(data.model_id)} &nbsp; <strong>Duration:</strong> ${data.duration}s`;
  resultMeta.innerHTML = metaHtml;

  // Audio player
  const audioUrl = data.url;
  initWaveSurfer(audioUrl);

  // Download link
  downloadLink.href = data.download_url;
  downloadLink.download = data.filename;

  // Show result card
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ============================================================
// Play / pause
// ============================================================
playPauseBtn.addEventListener("click", () => {
  if (!wavesurfer) return;
  wavesurfer.playPause();
});

volumeSlider.addEventListener("input", () => {
  if (wavesurfer) wavesurfer.setVolume(parseFloat(volumeSlider.value));
});

// ============================================================
// Generate another
// ============================================================
generateAnother.addEventListener("click", () => {
  resultCard.classList.add("hidden");
  if (wavesurfer) { wavesurfer.stop(); }
  inputCard.scrollIntoView({ behavior: "smooth" });
});

// ============================================================
// History
// ============================================================
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/files`);
    const data = await res.json();
    renderHistory(data.files);
  } catch {
    historyList.innerHTML = `<p class="empty-state">Could not load history.</p>`;
  }
}

function renderHistory(files) {
  if (!files || files.length === 0) {
    historyList.innerHTML = `<p class="empty-state">No files yet. Generate some music above!</p>`;
    return;
  }

  historyList.innerHTML = files.map((f) => {
    const size = formatBytes(f.size_bytes);
    const date = new Date(f.created_at * 1000).toLocaleString();
    return `
      <div class="history-item">
        <div style="flex:1;min-width:0">
          <div class="history-item-name">🎵 ${escHtml(f.filename)}</div>
          <div class="history-item-meta">${size} · ${date}</div>
        </div>
        <div class="history-item-actions">
          <button class="history-play-btn" title="Play" onclick="playHistoryItem('${escHtml(f.url)}')">▶</button>
          <a href="${escHtml(f.download_url)}" download="${escHtml(f.filename)}" class="btn btn-sm btn-outline history-dl-btn">⬇ Download</a>
          <button class="btn btn-sm btn-outline" style="color:var(--danger)" onclick="deleteHistoryItem('${escHtml(f.filename)}')">🗑</button>
        </div>
      </div>`;
  }).join("");
}

window.playHistoryItem = function(url) {
  initWaveSurfer(url);
  resultCard.classList.remove("hidden");
  resultMeta.innerHTML = `<strong>Playing:</strong> ${url}`;
  downloadLink.href = url.replace("/outputs/", "/api/download/");
  resultCard.scrollIntoView({ behavior: "smooth" });
};

window.deleteHistoryItem = async function(filename) {
  if (!confirm(`Delete ${filename}?`)) return;
  try {
    const res = await fetch(`${API_BASE}/api/files/${encodeURIComponent(filename)}`, { method: "DELETE" });
    if (!res.ok) throw new Error();
    showToast("File deleted.", "success");
    loadHistory();
  } catch {
    showToast("Could not delete file.", "error");
  }
};

refreshBtn.addEventListener("click", loadHistory);

// ============================================================
// UI helpers
// ============================================================
function setGenerating(generating) {
  generateBtn.disabled = generating;
  if (generating) {
    btnLabel.textContent = "Generating…";
    btnSpinner.classList.remove("hidden");
    progressSec.classList.remove("hidden");
    progressText.textContent = "Preparing models…";
  } else {
    btnLabel.textContent = "✨ Generate Music";
    btnSpinner.classList.add("hidden");
    progressSec.classList.add("hidden");
  }
}

let toastTimer;
function showToast(message, type = "info") {
  toast.textContent = message;
  toast.className = `toast ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.classList.add("hidden"); }, 5000);
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function escHtml(str) {
  const d = document.createElement("div");
  d.textContent = String(str || "");
  return d.innerHTML;
}

// ============================================================
// Init
// ============================================================
checkHealth();
loadHistory();
