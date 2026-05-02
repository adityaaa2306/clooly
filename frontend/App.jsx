// frontend/App.jsx - React overlay UI
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

const MAX_TRANSCRIPTS = 8;
const RECONNECT_DELAY_MS = 2000;

const statusMeta = {
  idle: { label: "Idle", color: "#8b949e" },
  listening: { label: "Listening", color: "#39d98a" },
  processing: { label: "Thinking", color: "#f7c948" },
  timeout: { label: "Timeout", color: "#ff9f43" },
  error: { label: "Error", color: "#ff5c7a" },
};

function debugLog(message, level) {
  if (window.cluelyDebugLog) {
    window.cluelyDebugLog(message, level);
  } else if (level === "error") {
    console.error(message);
  } else {
    console.log(message);
  }
}

function getWebSocketUrl() {
  try {
    if (window.electronAPI?.getBackendWSUrl) {
      return window.electronAPI.getBackendWSUrl();
    }
  } catch (error) {
    debugLog("[WS] Failed to read preload URL: " + error.message, "error");
  }

  return "ws://localhost:8001/ws";
}

function StatusPill({ connected, status }) {
  const meta = statusMeta[status] || statusMeta.idle;

  return (
    <div className="status-pill">
      <span className="status-dot" style={{ backgroundColor: meta.color }} />
      <span>{meta.label}</span>
      <span className="status-separator">/</span>
      <span className={connected ? "connected" : "disconnected"}>
        {connected ? "Connected" : "Disconnected"}
      </span>
    </div>
  );
}

function TranscriptPanel({ transcripts }) {
  const latest = transcripts[transcripts.length - 1];
  const panelRef = useRef(null);

  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [transcripts]);

  return (
    <section className="transcript-wrap">
      <div className="quote-mark">"</div>
      <div className="live-caption">
        {latest ? latest.text : "Waiting for live speech..."}
      </div>
      <div className="transcript-history" ref={panelRef}>
        {transcripts.slice(-4).map((item) => (
          <div className={item.is_final ? "history-row final" : "history-row"} key={item.id}>
            <span>{item.speakerLabel}</span>
            <p>{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function AnswerBox({ answer, status }) {
  const isProcessing = status === "processing";

  return (
    <section className="answer-shell">
      <div className="answer-header">
        <span>What should I say?</span>
        {isProcessing && <span className="typing-indicator">Streaming</span>}
      </div>
      <div className="answer-content">
        {answer || (isProcessing ? "Thinking..." : "Ask a question and I will draft a concise answer.")}
      </div>
    </section>
  );
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("idle");
  const [transcripts, setTranscripts] = useState([]);
  const [answer, setAnswer] = useState("");

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const shellRef = useRef(null);
  const lastResizeHeightRef = useRef(0);
  const wsUrl = useMemo(() => getWebSocketUrl(), []);

  useLayoutEffect(() => {
    const shell = shellRef.current;
    const resizeWindow = window.electronAPI?.resizeWindow;
    if (!shell || !resizeWindow) {
      return undefined;
    }

    let frameId = 0;
    const requestResize = () => {
      window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        const nextHeight = Math.ceil(shell.scrollHeight + 2);
        if (Math.abs(nextHeight - lastResizeHeightRef.current) > 4) {
          lastResizeHeightRef.current = nextHeight;
          resizeWindow(nextHeight);
        }
      });
    };

    requestResize();
    const observer = new ResizeObserver(requestResize);
    observer.observe(shell);

    return () => {
      window.cancelAnimationFrame(frameId);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    function scheduleReconnect() {
      if (cancelled) return;
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
    }

    function connect() {
      if (cancelled) return;

      debugLog("[WS] Connecting to " + wsUrl);
      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        reconnectAttemptsRef.current = 0;
        setConnected(true);
        setStatus((current) => (current === "idle" || current === "error" ? "listening" : current));
        debugLog("[WS] Connected");
      };

      socket.onmessage = (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch (error) {
          debugLog("[WS] Invalid JSON: " + error.message, "error");
          return;
        }

        if (message.type === "status") {
          setStatus(message.state || "idle");
          if (message.state === "processing") {
            setAnswer("");
          }
          return;
        }

        if (message.type === "transcript") {
          const speakerLabel =
            message.speaker === "interviewer"
              ? "Interviewer"
              : message.speaker === "user"
                ? "Candidate"
                : "Speaker";
          const isFinal = Boolean(message.is_final);

          setTranscripts((current) => {
            const nextItem = {
              id: Date.now() + Math.random(),
              speakerLabel,
              text: message.text || "",
              is_final: isFinal,
            };
            const next = [...current];
            const last = next[next.length - 1];

            if (last && !last.is_final && last.speakerLabel === speakerLabel) {
              next[next.length - 1] = { ...nextItem, id: last.id };
            } else {
              next.push(nextItem);
            }

            return next.slice(-MAX_TRANSCRIPTS);
          });
          return;
        }

        if (message.type === "answer") {
          if (message.chunk) {
            setAnswer((current) => current + (message.text || ""));
          } else {
            setStatus((current) => (current === "processing" ? "listening" : current));
          }
        }
      };

      socket.onerror = () => {
        setStatus("error");
        debugLog("[WS] Socket error", "error");
      };

      socket.onclose = () => {
        if (socketRef.current === socket) {
          socketRef.current = null;
        }

        setConnected(false);
        setStatus((current) => (current === "processing" ? "timeout" : "idle"));

        if (!cancelled) {
          reconnectAttemptsRef.current += 1;
          debugLog("[WS] Disconnected, retry " + reconnectAttemptsRef.current);
          scheduleReconnect();
        }
      };
    }

    connect();

    return () => {
      cancelled = true;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, [wsUrl]);

  return (
    <main className="overlay-shell" ref={shellRef}>
      <style>{`
        .overlay-shell {
          width: 100%;
          min-height: 100%;
          height: auto;
          padding: 14px 16px 16px;
          color: #151922;
          background:
            linear-gradient(135deg, rgba(248, 250, 252, 0.82), rgba(223, 228, 236, 0.66)),
            linear-gradient(90deg, rgba(255, 255, 255, 0.42), rgba(176, 186, 200, 0.22));
          backdrop-filter: blur(20px) saturate(1.25);
          border: 1px solid rgba(255, 255, 255, 0.72);
          border-radius: 16px;
          box-shadow: 0 22px 60px rgba(15, 23, 42, 0.24), inset 0 1px 0 rgba(255,255,255,0.72);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          -webkit-app-region: drag;
        }

        .top-bar {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          height: 28px;
        }

        .prompt-pill {
          padding: 7px 14px;
          border: 1px solid rgba(255, 255, 255, 0.18);
          border-radius: 14px;
          background: linear-gradient(180deg, #2c70d6, #155fc7);
          color: white;
          font-size: 14px;
          font-weight: 700;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 10px 24px rgba(21,95,199,0.28);
        }

        .status-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          color: rgba(21, 25, 34, 0.76);
          font-size: 12px;
          font-weight: 700;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          box-shadow: 0 0 14px currentColor;
        }

        .status-separator {
          color: rgba(21,25,34,0.32);
        }

        .connected {
          color: #057a45;
        }

        .disconnected {
          color: #b42318;
        }

        .transcript-wrap {
          position: relative;
          margin-top: 10px;
          min-height: 86px;
        }

        .quote-mark {
          position: absolute;
          left: -8px;
          top: -10px;
          color: rgba(21,25,34,0.22);
          font-size: 46px;
          line-height: 1;
        }

        .live-caption {
          min-height: 42px;
          max-height: 62px;
          overflow: hidden;
          padding-left: 8px;
          color: #181c25;
          font-size: 16px;
          font-weight: 650;
          line-height: 1.45;
        }

        .transcript-history {
          height: 42px;
          overflow: hidden;
          margin-top: 5px;
          padding-left: 8px;
        }

        .history-row {
          display: flex;
          gap: 8px;
          align-items: baseline;
          min-width: 0;
          margin-top: 2px;
          color: rgba(21, 25, 34, 0.52);
          font-size: 12px;
        }

        .history-row.final {
          color: rgba(21, 25, 34, 0.76);
        }

        .history-row span {
          flex: 0 0 auto;
          color: rgba(22, 94, 172, 0.92);
          font-weight: 700;
        }

        .history-row p {
          min-width: 0;
          margin: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .tool-row {
          display: flex;
          align-items: center;
          gap: 20px;
          height: 34px;
          margin-top: 4px;
          color: rgba(21, 25, 34, 0.72);
          border-bottom: 1px solid rgba(21,25,34,0.12);
        }

        .tool-item {
          display: inline-flex;
          gap: 8px;
          align-items: center;
          font-size: 14px;
          font-weight: 700;
        }

        .tool-dot {
          color: rgba(21,25,34,0.28);
        }

        .answer-shell {
          margin-top: 12px;
          border: 1px solid rgba(255,255,255,0.78);
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.42);
          overflow: hidden;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.65), 0 12px 26px rgba(15,23,42,0.12);
        }

        .answer-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 9px 12px;
          color: rgba(21, 25, 34, 0.68);
          border-bottom: 1px solid rgba(21,25,34,0.09);
          font-size: 13px;
          font-weight: 750;
        }

        .typing-indicator {
          color: #9a6700;
          font-size: 12px;
        }

        .answer-content {
          min-height: 96px;
          overflow: visible;
          padding: 11px 12px 12px;
          color: #10141d;
          font-size: 14px;
          line-height: 1.46;
          white-space: pre-wrap;
          word-break: break-word;
        }
      `}</style>

      <div className="top-bar">
        <div className="prompt-pill">What should I say?</div>
      </div>
      <StatusPill connected={connected} status={status} />
      <TranscriptPanel transcripts={transcripts} />
      <div className="tool-row">
        <span className="tool-item">Assist</span>
        <span className="tool-dot">-</span>
        <span className="tool-item">What should I say?</span>
        <span className="tool-dot">-</span>
        <span className="tool-item">Follow-up questions</span>
        <span className="tool-dot">-</span>
        <span className="tool-item">Recap</span>
      </div>
      <AnswerBox answer={answer} status={status} />
    </main>
  );
}
