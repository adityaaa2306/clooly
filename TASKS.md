# TASKS.md — Cluely MVP Implementation Checklist

Tasks are ordered by dependency. Autopilot must complete them **in sequence**.
Each task has: inputs, outputs, acceptance criteria, and the file it lives in.

---

## PHASE 1 — Foundation (No External APIs)

### TASK-01: Python Backend Entrypoint
**File:** `backend/main.py`
**Description:** FastAPI app with a single WebSocket endpoint at `/ws`.
**Inputs:** None
**Outputs:**
- `GET /health` → `{"status": "ok"}`
- `WS /ws` → accepts connections, echoes `{"type": "status", "state": "idle"}`
**Acceptance:** `uvicorn backend.main:app --port 8000` starts without error. `/health` returns 200.

---

### TASK-02: Electron Main Process
**File:** `electron/main.js`
**Description:** Electron app that:
1. Spawns the Python backend as a child process (`python -m uvicorn backend.main:app --port 8000`)
2. Waits for `/health` to return 200 before loading the renderer
3. Creates a **frameless, always-on-top, semi-transparent** BrowserWindow (800×200px)
4. Loads `frontend/index.html`
**Inputs:** Python process spawned on app `ready`
**Outputs:** Electron window opens. Python subprocess visible in process list.
**Acceptance:** Running `npm start` opens the overlay window.

---

### TASK-03: Electron Preload + React Bootstrap
**Files:** `electron/preload.js`, `frontend/index.html`, `frontend/index.jsx`
**Description:**
- `preload.js`: expose `window.electronAPI = { sendMessage, onMessage }` via `contextBridge`
- `index.html`: minimal HTML shell, loads `index.jsx` via bundler or script tag
- `index.jsx`: `ReactDOM.createRoot` mounting `<App />`
**Acceptance:** Electron window renders "Cluely Ready" text from React.

---

### TASK-04: Rolling Transcript Buffer
**File:** `backend/context.py`
**Class:** `ContextEngine`
**Methods:**
```python
def add_transcript(self, speaker: str, text: str, is_final: bool) -> None
def get_last_question(self) -> str | None   # returns latest interviewer question
def get_summary(self) -> str               # last 2 exchanges, max 100 words
def clear(self) -> None
```
**Rules:**
- Buffer holds last 60 seconds of final transcripts only
- Speaker label `"interviewer"` = diarization speaker 0
- A "question" = utterance ending with `?` OR confidence > 0.85 AND speaker = interviewer
**Acceptance:** Unit test: add 5 transcripts, `get_last_question()` returns the most recent one with `?`.

---

## PHASE 2 — External API Integration

### TASK-05: Deepgram Flux STT Client
**File:** `backend/stt.py`
**Class:** `DeepgramSTTClient`
**Description:** Connects to Deepgram Flux via streaming WebSocket.
**Config (from `.env`):**
```python
model         = DEEPGRAM_MODEL      # "flux-general-en"
smart_format  = True
interim_results = True
filler_words  = False
encoding      = "linear16"
sample_rate   = 16000
diarize       = True
endpointing   = 200                 # ms — Eager EOT
```
**Callbacks:**
```python
on_interim(text: str, speaker: int) -> None
on_final(text: str, speaker: int, confidence: float) -> None
on_eot() -> None    # triggers LLM pipeline
```
**Audio Source:** System default microphone via `pyaudio`, 16kHz, mono, 16-bit PCM.
**Acceptance:** Running `stt.py` directly prints live interim transcripts to stdout.

---

### TASK-06: Groq LLM Streaming Client
**File:** `backend/llm.py`
**Class:** `GroqLLMClient`
**Method:**
```python
async def stream_answer(self, question: str, context: str) -> AsyncGenerator[str, None]
```
**System prompt (exact, do not expand):**
```
You are a real-time interview copilot. Give concise, accurate answers in 3-5 bullet points. Max 150 words. No preamble.
```
**User prompt template:**
```
Context: {context}
Question: {question}
Answer:
```
**Config:**
- Model: `GROQ_MODEL` from env (default: `llama3-8b-8192`)
- `max_tokens`: `MAX_TOKENS` from env (default: `200`)
- `temperature`: `0.3`
- `stream`: `True`
**Acceptance:** `stream_answer("What is a binary tree?", "")` yields token chunks in < 800ms first token.

---

### TASK-07: Async Pipeline Orchestrator
**File:** `backend/pipeline.py`
**Class:** `CopilotPipeline`
**Description:** Wires STT → ContextEngine → LLM → WebSocket broadcast
**Flow:**
```
STT.on_final()  → context.add_transcript()
STT.on_eot()    → question = context.get_last_question()
                → if question: stream LLM answer over WebSocket
```
**Timing:** Log `perf_counter()` at:
1. EOT detected
2. LLM first token received
3. LLM stream complete
**WebSocket messages emitted:**
```json
{ "type": "transcript", "text": "...", "speaker": "interviewer", "is_final": true }
{ "type": "answer", "text": "...", "chunk": true }
{ "type": "answer", "text": "", "chunk": false }  ← signals stream end
{ "type": "status", "state": "listening" }
```
**Acceptance:** End-to-end pipeline runs. Logs show `LLM first token < 1100ms` from EOT.

---

## PHASE 3 — UI Layer

### TASK-08: React Overlay UI
**File:** `frontend/App.jsx`
**Description:** Minimal, always-on-top copilot UI.
**Components:**
```
<App>
  <StatusDot />            ← green=listening, yellow=processing, grey=idle
  <TranscriptPanel />      ← last 3 final transcript lines, auto-scroll
  <AnswerBox />            ← streams in tokens as they arrive, clears on new question
</App>
```
**WebSocket:** Connect to `ws://localhost:8000/ws` on mount. Reconnect on disconnect.
**Styling:**
- Background: `rgba(0,0,0,0.75)`, `backdrop-filter: blur(8px)`
- Font: monospace, white, 13px
- No scrollbars visible
- Window: 800px wide, auto-height, max 250px
**Acceptance:** UI renders live transcript and streams LLM answer tokens in real time.

---

## PHASE 4 — Latency Test (Acceptance Gate)

### TASK-09: End-to-End Latency Benchmark
**File:** `tests/test_latency.py`
**Description:** Simulates the full pipeline with a pre-recorded audio clip.
**Test steps:**
1. Start the pipeline with a 5-second WAV file (question: "What is a binary search tree?")
2. Inject audio into STT client
3. Record `t_eot` when EOT fires
4. Record `t_first_token` when first LLM chunk arrives
5. Record `t_complete` when stream ends
**Assertions:**
```python
assert (t_first_token - t_eot) < 1100   # ms: STT EOT → LLM first token
assert (t_complete - t_eot)   < 1500   # ms: total TTA
```
**Fixture audio:** generate a 5s sine-wave WAV with `scipy` if no real audio is available.
**Acceptance:** `pytest tests/test_latency.py` passes. Both assertions green.

---

## PHASE 5 — Polish + Packaging

### TASK-10: Graceful Shutdown + Error Handling
**Files:** `electron/main.js`, `backend/main.py`
**Description:**
- On Electron quit: kill Python subprocess cleanly
- On WebSocket disconnect: frontend retries every 2 seconds, max 5 retries
- On LLM timeout (>1200ms): emit `{"type": "status", "state": "timeout"}`, reset pipeline
- On STT error: log + reconnect Deepgram within 500ms

---

### TASK-11: README
**File:** `README.md`
**Must include:**
1. Prerequisites (Python 3.11, Node 18+, portaudio for pyaudio)
2. Setup: `pip install -r requirements.txt` + `npm install`
3. `.env` setup instructions
4. Run: `npm start`
5. Latency test: `pytest tests/test_latency.py`

---

## ✅ Definition of Done

The project is complete when:
- [ ] `npm start` opens the Electron overlay
- [ ] Speaking into the mic shows live transcript in the UI
- [ ] A question triggers an AI answer within 1.5 seconds
- [ ] `pytest tests/test_latency.py` passes
- [ ] No uncaught exceptions in a 5-minute live session