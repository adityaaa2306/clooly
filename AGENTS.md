# AGENTS.md — Cluely MVP: Real-Time Interview Copilot

## 🧠 Project Overview

Build a real-time AI interview copilot as an **Electron desktop app**.
It listens to live audio, transcribes via Deepgram Flux, detects when a question ends,
and generates a short AI answer in **≤1.5 seconds total end-to-end**.

---

## 🏗️ Tech Stack (Non-Negotiable)

| Layer            | Technology                                         |
|------------------|----------------------------------------------------|
| Desktop Shell    | Electron (Node.js)                                 |
| Frontend UI      | React (JSX) via Electron renderer                  |
| Backend Runtime  | Python 3.11+                                       |
| Backend Server   | FastAPI + Uvicorn (WebSocket)                      |
| STT Provider     | Deepgram Flux (`flux-general-en`) — streaming      |
| LLM Provider     | NVIDIA NIM API — `openai/gpt-oss-20b`              |
| LLM Client       | `openai` Python SDK (OpenAI-compatible base_url)   |
| IPC              | Electron ↔ Python via WebSocket on localhost       |
| Audio Capture    | `sounddevice` (system mic, 16kHz mono 16-bit PCM)      |

---
    ## ⚙️ Python Version
    - Required: Python 3.12.x
    - Do NOT use Python 3.13 or 3.14 — pre-built wheels for aiohttp and pydantic-core do not exist yet
    - Always create venv with: python -m venv .venv (not py -m venv)
    ---

## 📁 Exact File/Folder Structure

```
cluely-mvp/
├── backend/
│   ├── __init__.py         # empty
│   ├── context.py          # rolling transcript buffer + question detection
│   ├── llm.py              # NVIDIA NIM streaming inference client
│   ├── main.py             # FastAPI app + WebSocket endpoint
│   ├── pipeline.py         # async orchestrator: STT → context → LLM
│   └── stt.py              # Deepgram Flux streaming client
├── electron/
│   ├── main.js             # Electron main process: creates BrowserWindow, spawns Python
│   └── preload.js          # contextBridge: exposes ipcRenderer to React
├── frontend/
│   ├── App.jsx             # React root: transcript panel + answer box
│   ├── index.html          # HTML shell loaded by Electron
│   └── index.jsx           # ReactDOM.createRoot entry point
├── tests/
│   └── test_latency.py     # End-to-end TTA benchmark (MUST pass ≤1500ms)
├── .env                    # local secrets (never commit)
├── .env.example            # template for all required env vars
├── .gitignore
├── AGENTS.md               # this file
├── TASKS.md                # ordered implementation checklist
├── package.json            # Electron + React deps
├── README.md
└── requirements.txt        # Python deps
```

---

## ⚙️ Environment Variables

All secrets live in `.env`. Load in Python with `python-dotenv`.

| Variable                  | Example Value                           |
|---------------------------|-----------------------------------------|
| `DEEPGRAM_API_KEY`        | `dg_xxxx`                               |
| `DEEPGRAM_MODEL`          | `flux-general-en`                       |
| `DEEPGRAM_ENDPOINTING_MS` | `200`                                   |
| `NVIDIA_NIM_API_KEY`      | `nvapi-xxxx`                            |
| `NVIDIA_NIM_BASE_URL`     | `https://integrate.api.nvidia.com/v1`   |
| `NIM_MODEL`               | `openai/gpt-oss-20b`                    |
| `NIM_TEMPERATURE`         | `0.6`                                   |
| `NIM_TOP_P`               | `0.9`                                   |
| `MAX_TOKENS`              | `200`                                   |
| `BACKEND_HOST`            | `localhost`                             |
| `BACKEND_PORT`            | `8000`                                  |
| `CONTEXT_WINDOW_SECONDS`  | `45`                                    |
| `AUDIO_SAMPLE_RATE`       | `16000`                                 |

---

## 🎤 Deepgram Flux — Exact Connection Config

```python
# backend/stt.py — use exactly these parameters
with client.listen.v2.connect(
    model="flux-general-en",       # Flux reasoning model, not Nova
    smart_format=True,             # Auto punctuation + question marks
    interim_results=True,          # Stream text as spoken (low latency)
    filler_words=True,             # Pass "um/uh" to aid flow detection
    encoding="linear16",
    sample_rate=16000,
    diarize=True,                  # Speaker 0 = interviewer, Speaker 1 = user
    endpointing=200,               # ms — eager end-of-turn detection
) as connection:
```

**Why Flux over Nova-3:**
- Re-evaluates previous words as context arrives (corrects mid-sentence errors)
- Higher EOT confidence = precise LLM trigger timing
- Keyterm prompting supports domain words like "Kubernetes", "LeetCode", "Big-O"

---

## 🤖 NVIDIA NIM LLM — Exact Client Code

```python
# backend/llm.py — use exactly this client setup
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
)

completion = client.chat.completions.create(
    model=os.getenv("NIM_MODEL", "openai/gpt-oss-20b"),
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=float(os.getenv("NIM_TEMPERATURE", "0.6")),
    top_p=float(os.getenv("NIM_TOP_P", "0.9")),
    max_tokens=int(os.getenv("MAX_TOKENS", "200")),
    stream=True,
)

# Stream tokens — skip reasoning_content, yield only content
for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        yield delta.content   # send each token over WebSocket
```

**Note:** GPT OSS 20B may emit `reasoning_content` chunks (chain-of-thought).
**Ignore these** — only yield `delta.content` tokens to the UI.

---

## 🔌 Inter-Process Communication (IPC)

```
[Electron Main Process]
    ↓ spawns subprocess
[Python FastAPI backend] ← runs on localhost:8000
    ↓ WebSocket ws://localhost:8000/ws
[Electron Renderer (React)]
    ↓ receives {type, payload} JSON messages
[UI updates in <50ms]
```

WebSocket message schema (backend → frontend):
```json
{ "type": "transcript", "text": "...", "speaker": "interviewer", "is_final": true }
{ "type": "answer",     "text": "...", "chunk": true }
{ "type": "answer",     "text": "",   "chunk": false }
{ "type": "status",     "state": "listening" | "processing" | "idle" | "timeout" }
```

---

## ⏱️ Latency Budget (Hard Constraint)

| Stage                        | Target      | Failure Threshold |
|------------------------------|-------------|-------------------|
| STT (Deepgram Flux)          | 200–400ms   | >600ms            |
| EOT Detection (endpointing)  | ~200ms      | >300ms            |
| LLM Inference (NIM)          | 500–1000ms  | >1200ms           |
| UI Render (Electron)         | ~50ms       | >100ms            |
| **Total End-to-End**         | **≤1.5s**   | **>2.0s = FAIL**  |

`tests/test_latency.py` must assert `TTA ≤ 1500ms`. If it fails, the implementation is wrong.

---

## 🔑 Implementation Rules for Autopilot

1. **Never block the async event loop.** All I/O in `backend/` must be `async/await`.
2. **Stream everything.** `interim_results=True` in Deepgram. Stream tokens from NIM.
3. **Trigger LLM only on EOT.** Never call NIM on interim transcripts.
4. **Skip `reasoning_content` chunks.** Only stream `delta.content` to the UI.
5. **Keep prompts minimal.** System prompt ≤ 50 tokens. User prompt = last question + 2-line context.
6. **Max 200 tokens from LLM.** Short answers only.
7. **Electron window: frameless, always-on-top, semi-transparent.**
8. **Python process must start before renderer loads.** Health-check loop in `electron/main.js`.
9. **Log all timings** with `time.perf_counter()` at each pipeline stage.
10. **`test_latency.py` is the acceptance gate.** All tasks are incomplete until it passes.

---

## 🚫 Out of Scope (Do Not Implement)

- User authentication
- Database / persistent storage
- Multi-language support
- CRM integrations
- Long-term memory
- Screen capture / OCR