"""
Meeting & Voicemail Transcription Server
Run: uvicorn transcribe_app:app --host 0.0.0.0 --port 8000
Then open: http://localhost:8000  (or your Tailscale IP)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio, tempfile, os, shutil

app = FastAPI()

# ── Whisper model (loaded once at startup) ──────────────────────────────────
from faster_whisper import WhisperModel

print("Loading Whisper model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model ready.")

# ── Transcription helper ────────────────────────────────────────────────────
def transcribe_file(path: str) -> dict:
    segments, info = model.transcribe(path, beam_size=5)
    full_text = " ".join(seg.text.strip() for seg in segments)
    return {
        "text": full_text,
        "language": info.language,
        "duration": round(info.duration, 1),
    }

# ── API endpoint ────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = await asyncio.to_thread(transcribe_file, tmp_path)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

# ── Serve UI ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

# ── UI (embedded) ───────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcribe</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d0f12;
    --surface:  #151820;
    --border:   #232730;
    --accent:   #00e5a0;
    --accent2:  #0099ff;
    --danger:   #ff4757;
    --text:     #e2e8f0;
    --muted:    #5a6478;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,229,160,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,160,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  header {
    width: 100%;
    max-width: 720px;
    margin-bottom: 40px;
  }

  .logo {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
  }

  h1 {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -.02em;
    line-height: 1.2;
  }

  h1 span { color: var(--accent); }

  .subtitle {
    margin-top: 8px;
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
  }

  /* Tabs */
  .tabs {
    width: 100%;
    max-width: 720px;
    display: flex;
    gap: 2px;
    margin-bottom: 2px;
  }

  .tab {
    flex: 1;
    padding: 12px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: .1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .15s;
    border-bottom: none;
  }

  .tab:first-child { border-radius: 6px 0 0 0; }
  .tab:last-child  { border-radius: 0 6px 0 0; }

  .tab.active {
    background: var(--surface);
    color: var(--accent);
    border-color: var(--accent);
    border-bottom-color: var(--surface);
  }

  /* Card */
  .card {
    width: 100%;
    max-width: 720px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0 6px 6px 6px;
    padding: 32px;
  }

  .panel { display: none; }
  .panel.active { display: block; }

  /* Record panel */
  .record-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 24px;
    padding: 16px 0;
  }

  .rec-btn {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: transparent;
    cursor: pointer;
    position: relative;
    transition: all .2s;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .rec-btn::before {
    content: '';
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--accent);
    transition: all .2s;
  }

  .rec-btn:hover { box-shadow: 0 0 24px rgba(0,229,160,.2); }

  .rec-btn.recording {
    border-color: var(--danger);
    animation: pulse 1.4s ease infinite;
  }

  .rec-btn.recording::before {
    background: var(--danger);
    border-radius: 6px;
    width: 30px;
    height: 30px;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,.4); }
    50%       { box-shadow: 0 0 0 16px rgba(255,71,87,0); }
  }

  .timer {
    font-family: var(--mono);
    font-size: 32px;
    letter-spacing: .1em;
    color: var(--text);
  }

  .timer.recording { color: var(--danger); }

  .rec-status {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .rec-status.recording { color: var(--danger); }

  /* Visualizer */
  canvas {
    width: 100%;
    height: 48px;
    border-radius: 4px;
    opacity: 0;
    transition: opacity .3s;
  }
  canvas.visible { opacity: 1; }

  /* Upload panel */
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    position: relative;
  }

  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent2);
    background: rgba(0,153,255,.04);
  }

  .dropzone input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }

  .dz-icon {
    font-size: 36px;
    margin-bottom: 12px;
  }

  .dz-label {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    margin-bottom: 6px;
  }

  .dz-sub {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: .05em;
  }

  .file-selected {
    margin-top: 16px;
    padding: 12px 16px;
    background: rgba(0,153,255,.08);
    border: 1px solid rgba(0,153,255,.25);
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent2);
    display: none;
  }

  /* Shared transcribe button */
  .transcribe-btn {
    width: 100%;
    margin-top: 24px;
    padding: 14px;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: #000;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .15s;
  }

  .transcribe-btn:hover:not(:disabled) {
    background: #00ffb3;
    box-shadow: 0 0 20px rgba(0,229,160,.3);
  }

  .transcribe-btn:disabled {
    opacity: .35;
    cursor: not-allowed;
  }

  /* Output */
  .output {
    margin-top: 32px;
    display: none;
  }

  .output.visible { display: block; }

  .output-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .output-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--accent);
  }

  .meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }

  .copy-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: all .15s;
    letter-spacing: .08em;
  }

  .copy-btn:hover { border-color: var(--accent); color: var(--accent); }

  .transcript-box {
    background: #0a0c0f;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
    font-size: 14px;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
    min-height: 80px;
    font-weight: 300;
  }

  /* Spinner / progress */
  .progress-wrap {
    margin-top: 24px;
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }

  .progress-wrap.visible { display: flex; }

  .spinner {
    width: 32px;
    height: 32px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .progress-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .15em;
    color: var(--muted);
    text-transform: uppercase;
  }

  .divider {
    height: 1px;
    background: var(--border);
    margin: 28px 0;
  }

  footer {
    margin-top: 32px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: .08em;
  }
</style>
</head>
<body>

<header>
  <div class="logo">RealDeal CPA // Tools</div>
  <h1>Audio <span>Transcribe</span></h1>
  <p class="subtitle">Record a meeting or upload a voicemail — powered by Whisper large-v3 on local GPU</p>
</header>

<div class="tabs">
  <button class="tab active" onclick="switchTab('record', this)">⏺ &nbsp;Record Meeting</button>
  <button class="tab" onclick="switchTab('upload', this)">⬆ &nbsp;Upload File</button>
</div>

<div class="card">

  <!-- RECORD PANEL -->
  <div id="panel-record" class="panel active">
    <div class="record-center">
      <button class="rec-btn" id="recBtn" onclick="toggleRecord()"></button>
      <div class="timer" id="timer">00:00</div>
      <div class="rec-status" id="recStatus">Click to start recording</div>
      <canvas id="visualizer"></canvas>
    </div>
    <div class="divider"></div>
    <button class="transcribe-btn" id="transcribeRecBtn" disabled onclick="transcribeRecording()">
      Transcribe Recording
    </button>
  </div>

  <!-- UPLOAD PANEL -->
  <div id="panel-upload" class="panel">
    <div class="dropzone" id="dropzone">
      <input type="file" id="fileInput" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm" onchange="onFileSelect(event)">
      <div class="dz-icon">🎙</div>
      <div class="dz-label">Drop audio file here or click to browse</div>
      <div class="dz-sub">WAV · MP3 · M4A · OGG · FLAC · WEBM</div>
    </div>
    <div class="file-selected" id="fileSelected"></div>
    <button class="transcribe-btn" id="transcribeUploadBtn" disabled onclick="transcribeUpload()">
      Transcribe File
    </button>
  </div>

  <!-- SHARED OUTPUT -->
  <div class="progress-wrap" id="progressWrap">
    <div class="spinner"></div>
    <div class="progress-label" id="progressLabel">Transcribing...</div>
  </div>

  <div class="output" id="output">
    <div class="output-header">
      <span class="output-label">Transcript</span>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="meta" id="metaInfo"></span>
        <button class="copy-btn" onclick="copyText()">Copy</button>
      </div>
    </div>
    <div class="transcript-box" id="transcriptBox"></div>
  </div>

</div>

<footer>Whisper large-v3 · CUDA · Local inference · No data leaves this machine</footer>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}

// ── Recording ──────────────────────────────────────────────────────────────
let mediaRecorder, audioChunks = [], timerInterval, elapsed = 0;
let audioContext, analyser, animFrame, recordedBlob;

async function toggleRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  audioContext.createMediaStreamSource(stream).connect(analyser);

  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  audioChunks = [];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
    document.getElementById('transcribeRecBtn').disabled = false;
  };
  mediaRecorder.start();

  elapsed = 0;
  timerInterval = setInterval(() => {
    elapsed++;
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('timer').textContent = `${m}:${s}`;
  }, 1000);

  document.getElementById('recBtn').classList.add('recording');
  document.getElementById('timer').classList.add('recording');
  document.getElementById('recStatus').textContent = 'Recording...';
  document.getElementById('recStatus').classList.add('recording');
  document.getElementById('transcribeRecBtn').disabled = true;
  drawVisualizer();
}

function stopRecording() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  clearInterval(timerInterval);
  cancelAnimationFrame(animFrame);

  document.getElementById('recBtn').classList.remove('recording');
  document.getElementById('timer').classList.remove('recording');
  document.getElementById('recStatus').textContent = 'Recording stopped — ready to transcribe';
  document.getElementById('recStatus').classList.remove('recording');
  document.getElementById('visualizer').classList.remove('visible');
}

function drawVisualizer() {
  const canvas = document.getElementById('visualizer');
  canvas.classList.add('visible');
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(analyser.frequencyBinCount);

  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(buf);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barW = canvas.width / buf.length * 2.5;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const h = (buf[i] / 255) * canvas.height;
      const alpha = 0.4 + (buf[i] / 255) * 0.6;
      ctx.fillStyle = `rgba(0,229,160,${alpha})`;
      ctx.fillRect(x, canvas.height - h, barW - 1, h);
      x += barW;
    }
  }
  draw();
}

async function transcribeRecording() {
  if (!recordedBlob) return;
  const file = new File([recordedBlob], 'recording.webm', { type: 'audio/webm' });
  await doTranscribe(file);
}

// ── Upload ─────────────────────────────────────────────────────────────────
let selectedFile = null;

function onFileSelect(e) {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;
  const el = document.getElementById('fileSelected');
  el.style.display = 'block';
  el.textContent = `📎  ${selectedFile.name}  (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`;
  document.getElementById('transcribeUploadBtn').disabled = false;
}

const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) {
    selectedFile = f;
    const el = document.getElementById('fileSelected');
    el.style.display = 'block';
    el.textContent = `📎  ${f.name}  (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
    document.getElementById('transcribeUploadBtn').disabled = false;
  }
});

async function transcribeUpload() {
  if (!selectedFile) return;
  await doTranscribe(selectedFile);
}

// ── Core transcribe ────────────────────────────────────────────────────────
async function doTranscribe(file) {
  const progress = document.getElementById('progressWrap');
  const output   = document.getElementById('output');
  const label    = document.getElementById('progressLabel');

  output.classList.remove('visible');
  progress.classList.add('visible');

  const msgs = ['Sending to Whisper...', 'Processing audio...', 'Generating transcript...'];
  let mi = 0;
  label.textContent = msgs[mi];
  const msgTimer = setInterval(() => { mi = (mi + 1) % msgs.length; label.textContent = msgs[mi]; }, 2500);

  try {
    const form = new FormData();
    form.append('file', file);

    const res = await fetch('/transcribe', { method: 'POST', body: form });
    clearInterval(msgTimer);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();
    document.getElementById('transcriptBox').textContent = data.text;
    document.getElementById('metaInfo').textContent =
      `${data.language.toUpperCase()} · ${Math.floor(data.duration / 60)}m ${Math.floor(data.duration % 60)}s`;
    progress.classList.remove('visible');
    output.classList.add('visible');

  } catch (err) {
    clearInterval(msgTimer);
    progress.classList.remove('visible');
    document.getElementById('transcriptBox').textContent = `Error: ${err.message}`;
    document.getElementById('output').classList.add('visible');
  }
}

// ── Copy ───────────────────────────────────────────────────────────────────
function copyText() {
  const text = document.getElementById('transcriptBox').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1800);
  });
}
</script>
</body>
</html>
"""
