import asyncio
import logging
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.context import ContextEngine
from backend.pipeline import CopilotPipeline
from backend.stt import audio_generator_from_microphone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cluely MVP Backend")

# CORS middleware to allow Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected WebSocket clients for broadcasting
connected_clients: set = set()
pipeline_task: Optional[asyncio.Task] = None
pipeline: Optional[CopilotPipeline] = None


async def _run_live_pipeline() -> None:
    """Run microphone -> Deepgram -> NIM -> WebSocket until cancelled."""
    global pipeline

    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    window_seconds = int(os.getenv("CONTEXT_WINDOW_SECONDS", "45"))

    context_engine = ContextEngine(window_seconds=window_seconds)
    pipeline = CopilotPipeline(
        context_engine=context_engine,
        on_message=broadcast_message,
    )

    try:
        logger.info("Starting live audio pipeline")
        audio_stream = audio_generator_from_microphone(sample_rate=sample_rate)
        await pipeline.run_with_audio_stream(audio_stream)
    except asyncio.CancelledError:
        logger.info("Live audio pipeline cancelled")
        raise
    except Exception:
        logger.exception("Live audio pipeline failed")
        await broadcast_message({"type": "status", "state": "error"})
    finally:
        pipeline = None
        logger.info("Live audio pipeline stopped")


async def ensure_pipeline_running() -> None:
    """Start the live pipeline once at least one renderer is connected."""
    global pipeline_task

    if pipeline_task and not pipeline_task.done():
        return

    pipeline_task = asyncio.create_task(_run_live_pipeline())


async def stop_pipeline_if_idle() -> None:
    """Stop microphone capture when no renderers are connected."""
    global pipeline_task

    if connected_clients or not pipeline_task or pipeline_task.done():
        return

    logger.info("No WebSocket clients remain; stopping live pipeline")
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass
    finally:
        pipeline_task = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with Electron frontend."""
    await websocket.accept()
    logger.info("WebSocket client connected")
    connected_clients.add(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "state": "listening"
        })
        await ensure_pipeline_running()
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (if any)
                # This will block until the client sends a message or disconnects
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                logger.info(f"Received from client: {data}")
                
            except asyncio.TimeoutError:
                # Send periodic keepalive
                try:
                    await websocket.send_json({
                        "type": "status",
                        "state": "listening"
                    })
                except Exception as e:
                    logger.warning(f"Keepalive send failed: {e}")
                    break
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        connected_clients.discard(websocket)
        logger.info(f"WebSocket removed (active: {len(connected_clients)})")
        await stop_pipeline_if_idle()


async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    if not connected_clients:
        logger.debug(f"No clients connected, dropping message: {message['type']}")
        return
    
    disconnected = set()
    
    for client in list(connected_clients):
        try:
            await client.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            disconnected.add(client)
    
    # Remove disconnected clients
    for client in disconnected:
        connected_clients.discard(client)


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    logger.info("Backend startup - pipeline ready for connections")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Backend shutting down")
    global pipeline_task

    if pipeline_task and not pipeline_task.done():
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
        finally:
            pipeline_task = None

    # Close all WebSocket connections
    for client in list(connected_clients):
        await client.close()
    connected_clients.clear()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("BACKEND_HOST", "localhost"),
        port=int(os.getenv("BACKEND_PORT", "8001")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
