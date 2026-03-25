# 🎙 transcribe_app

> Local GPU-powered audio transcription. Record meetings live or drop in a voicemail — get clean text back in seconds. No cloud. No subscriptions. No data leaving your machine.

---

## What It Does

A single-file FastAPI server that spins up a browser UI and a `/transcribe` endpoint. Under the hood it runs **Whisper large-v3** on your local GPU via `faster-whisper`. Two modes:

| Mode | Use Case |
|------|----------|
| **Record Meeting** | Live mic capture with waveform visualizer + stopwatch |
| **Upload File** | Drag-and-drop any audio file for transcription |

Transcript drops with detected language and audio duration. One-click copy.

---

## Stack

```
FastAPI          → HTTP server + REST endpoint
faster-whisper   → Whisper large-v3 via CTranslate2 (CUDA)
Web Browser      → MediaRecorder API for live capture
uvicorn          → ASGI server
```

Everything runs locally. The UI is embedded directly in the Python file — no frontend build step, no Node, no webpack nonsense.

---

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA (tested on RTX 6000)
- CUDA 11.x or 12.x + cuDNN installed
- ~3 GB VRAM for `large-v3` (use `medium` if tight)

---

## Installation

```bash
# Clone or drop the files in a folder
pip install -r requirements.txt
```

**requirements.txt**
```
fastapi
uvicorn[standard]
faster-whisper
python-multipart
```

---

## Running

```bash
uvicorn transcribe_app:app --host 0.0.0.0 --port 8000
```

Open your browser to:

```
http://localhost:8000
```

Or from any device on your Tailscale network:

```
http://<your-tailscale-ip>:8000
```

> **First boot takes 30–60 seconds** while `large-v3` loads into GPU memory. Subsequent transcriptions are fast.

---

## Model Options

Edit this line in `transcribe_app.py` to trade speed for accuracy:

```python
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~1 GB | ⚡⚡⚡⚡ | ★★☆☆ |
| `base` | ~1 GB | ⚡⚡⚡⚡ | ★★★☆ |
| `medium` | ~2 GB | ⚡⚡⚡ | ★★★★ |
| `large-v3` | ~3 GB | ⚡⚡ | ★★★★★ |

For voicemails and meetings with background noise, `large-v3` is worth it.

---

## Supported Audio Formats

```
WAV  ·  MP3  ·  M4A  ·  OGG  ·  FLAC  ·  WEBM
```

Browser recordings are captured as `.webm` (Opus codec) — Whisper handles it natively.

---

## API

If you want to call it programmatically from n8n, Python, or anything else:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@voicemail.wav"
```

**Response:**
```json
{
  "text": "Hey this is John calling about the Q1 returns...",
  "language": "en",
  "duration": 47.3
}
```

---

## Running as a Background Service (optional)

To keep it alive after you close the terminal:

```bash
# Using screen
screen -S transcribe
uvicorn transcribe_app:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+A then D
# Reattach: screen -r transcribe
```

Or create a systemd service if you want it to start on boot.

---

## Tailscale Access

Since this binds to `0.0.0.0`, any device on your Tailscale network can reach it. No port forwarding. No firewall rules. Just:

```
http://<machine-tailscale-ip>:8000
```

Useful for transcribing voicemails from your phone while your GPU machine does the heavy lifting.

---

## Privacy

- ✅ 100% local inference — nothing is sent to OpenAI or any external API
- ✅ Temp files are created and immediately deleted after each transcription
- ✅ No logging of transcript content
- ✅ No accounts, no API keys, no telemetry

---

## Extending This

A few directions if you want to build on it:

- **Auto-save to Notion** — POST the transcript JSON to your Notion MCP after transcription
- **Speaker diarization** — add `pyannote.audio` for multi-speaker meeting notes
- **Summarization** — pipe the transcript to your local Ollama instance for a summary
- **n8n webhook** — hit a webhook after transcription to trigger downstream workflows

---

## License

Do whatever you want with it. It's yours.
