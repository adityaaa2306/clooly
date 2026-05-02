# Cluely MVP — Real-Time AI Interview Copilot

A real-time AI interview copilot built as an Electron desktop overlay. Listens to live audio, transcribes via **Deepgram Flux**, detects question endings, and generates AI answers in **≤1.5 seconds end-to-end**.

## 🎯 Features

- **Live Transcription**: Deepgram Flux streaming STT with speaker diarization
- **Smart Question Detection**: Automatically detects interview questions
- **Real-Time AI Answers**: NVIDIA NIM LLM (GPT OSS 20B) streaming responses
- **Overlay UI**: Always-on-top frameless Electron window with live transcript & answer streaming
- **Sub-1.5s Latency**: Optimized for interview response time ≤1500ms end-to-end
- **Graceful Error Handling**: Automatic reconnection, timeouts, health monitoring

## 📋 Prerequisites

### System Requirements
- **Python**: 3.12.x (NOT 3.13+)
- **Node.js**: 18+
- **PortAudio**: Required for microphone access
  - **Windows**: `choco install portaudio` (via Chocolatey) OR download from [PortAudio site](http://www.portaudio.com)
  - **macOS**: `brew install portaudio`
  - **Linux**: `apt-get install portaudio19-dev`

### API Keys Required
- **Deepgram API Key** → Sign up at [deepgram.com](https://deepgram.com)
- **NVIDIA NIM API Key** → Sign up at [nvidia.com/nim](https://www.nvidia.com/en-us/ai-data-science/products/api/)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd cluely-mvp
```

### 2. Create Python Virtual Environment

```bash
# IMPORTANT: Use Python 3.12.x, NOT 3.13+
python -m venv .venv

# Activate venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Node packages
npm install
```

### 4. Configure Environment

Create `.env` file in the root directory (copy from `.env.example`):

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

```
# .env
DEEPGRAM_API_KEY=your_deepgram_key_here
NVIDIA_NIM_API_KEY=your_nvidia_nim_key_here

# Other configs (usually default)
DEEPGRAM_MODEL=flux-general-en
DEEPGRAM_ENDPOINTING_MS=200
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=openai/gpt-oss-20b
NIM_TEMPERATURE=0.6
NIM_TOP_P=0.9
MAX_TOKENS=200
```

### 5. Run the Application

```bash
npm start
```

This will:
1. Spawn the Python FastAPI backend (port 8000)
2. Wait for health check to pass
3. Open the Electron overlay window

## 🧪 Testing

### Run Latency Test

```bash
# From the project root, with venv activated
pytest tests/test_latency.py -v -s
```

**Expected Output:**
```
✅ ASSERTION 1: Time to First Token (TTFT)
   ✓ XXXms < 1100ms ✓

✅ ASSERTION 2: Total Time to Answer (TTA)
   ✓ XXXms < 1500ms ✓

🎉 LATENCY TEST PASSED - All assertions green!
```

## 📁 Project Structure

```
cluely-mvp/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # FastAPI app + WebSocket /ws
│   ├── stt.py                 # Deepgram Flux streaming client
│   ├── llm.py                 # NVIDIA NIM LLM client
│   ├── context.py             # Rolling transcript buffer + question detection
│   ├── pipeline.py            # Async orchestrator (STT → Context → LLM)
│   └── __init__.py
├── electron/                   # Electron main process
│   ├── main.js                # App launcher, Python spawn, health checks
│   └── preload.js             # IPC bridge via contextBridge
├── frontend/                   # React UI
│   ├── App.jsx                # Main overlay component
│   ├── index.jsx              # React entry point
│   └── index.html             # HTML shell
├── tests/                      # Test suite
│   └── test_latency.py        # End-to-end latency benchmark
├── .env                       # Environment variables (DO NOT COMMIT)
├── .env.example               # Environment template
├── package.json               # Node.js dependencies
├── requirements.txt           # Python dependencies
├── AGENTS.md                  # Non-negotiable constraints
├── TASKS.md                   # Implementation tasks
└── README.md                  # This file
```

## 🔌 WebSocket Protocol

### Backend → Frontend Messages

The backend sends JSON messages over WebSocket (`ws://localhost:8000/ws`):

#### Transcript Event
```json
{
  "type": "transcript",
  "text": "What is a binary search tree?",
  "speaker": "interviewer",
  "is_final": true
}
```

#### Answer Stream Chunk
```json
{
  "type": "answer",
  "text": "A ",
  "chunk": true
}
```

#### End of Answer Stream
```json
{
  "type": "answer",
  "text": "",
  "chunk": false
}
```

#### Status Update
```json
{
  "type": "status",
  "state": "listening|processing|idle|error|timeout"
}
```

## ⏱️ Latency Targets (Hard Constraints)

| Stage | Target | Failure Threshold |
|-------|--------|-------------------|
| STT (Deepgram Flux) | 200-400ms | >600ms |
| EOT Detection | ~200ms | >300ms |
| LLM Inference (NIM) | 500-1000ms | >1200ms |
| UI Render (Electron) | ~50ms | >100ms |
| **Total End-to-End (TTA)** | **≤1.5s** | **>2.0s = FAIL** |

See `tests/test_latency.py` for acceptance gate.

## 🔧 Development

### Running in Dev Mode

```bash
npm run dev
```

This runs esbuild in watch mode alongside Electron for live reload.

### Debugging

#### Python Backend Logs
The Python process logs to stdout. Check Electron console output.

#### Frontend DevTools
Uncomment in `electron/main.js`:
```javascript
mainWindow.webContents.openDevTools();
```

#### Enable Debug Logging
```bash
# In .env:
LOG_LEVEL=DEBUG
DEBUG=true
```

## 🛑 Troubleshooting

### "ModuleNotFoundError: No module named 'backend'"
Make sure you're in the project root directory (`d:\cluely-mvp\cluely-mvp`) when running `npm start`.

### "Port 8000 already in use"
Kill the existing process:
```bash
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

### "Health check failed"
1. Check Python backend is running (visible in Electron logs)
2. Verify port 8000 is accessible
3. Check `.env` is loaded correctly

### "WebSocket disconnected" in UI
The frontend auto-reconnects every 2 seconds. Check:
1. Backend is running (`curl http://localhost:8000/health`)
2. Firewall allows localhost:8000

### "No microphone audio"
1. Check system microphone is enabled and selected
2. Install PortAudio (see Prerequisites)
3. Verify Deepgram API key is valid

## 🚀 Deployment

For production builds:

```bash
# Build React bundle
npm run build

# Package Electron app
npm run build:dist
```

## 📝 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | (required) | Deepgram API key |
| `DEEPGRAM_MODEL` | `flux-general-en` | STT model (must be Flux for reasoning) |
| `DEEPGRAM_ENDPOINTING_MS` | `200` | EOT detection threshold (ms) |
| `NVIDIA_NIM_API_KEY` | (required) | NVIDIA NIM API key |
| `NVIDIA_NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM API base URL |
| `NIM_MODEL` | `openai/gpt-oss-20b` | LLM model name |
| `NIM_TEMPERATURE` | `0.6` | Sampling temperature |
| `NIM_TOP_P` | `0.9` | Nucleus sampling |
| `MAX_TOKENS` | `200` | Max response tokens |
| `BACKEND_HOST` | `localhost` | Backend server host |
| `BACKEND_PORT` | `8000` | Backend server port |
| `CONTEXT_WINDOW_SECONDS` | `45` | Transcript buffer window (seconds) |
| `AUDIO_SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `LOG_LEVEL` | `INFO` | Logging level |

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please:
1. Create a feature branch
2. Test with `pytest tests/test_latency.py`
3. Ensure latency goals are met

## 📞 Support

For issues:
1. Check troubleshooting section above
2. Review backend logs (Electron console)
3. Verify `.env` configuration
4. Check API keys are valid

---

**Built for Real-Time Interview Excellence** ⚡
