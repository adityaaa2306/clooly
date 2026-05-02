"""Deepgram Flux streaming STT client."""

import asyncio
import audioop
import logging
import os
import time
from typing import Callable, Optional

from deepgram import AsyncDeepgramClient
from deepgram.core.api_error import ApiError

logger = logging.getLogger(__name__)


class DeepgramSTTClient:
    """Deepgram Flux v2 streaming STT client with interim, final, and EOT callbacks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "flux-general-en",
        endpointing_ms: int = 200,
        on_interim: Optional[Callable[[str, int], None]] = None,
        on_final: Optional[Callable[[str, int, float], None]] = None,
        on_eot: Optional[Callable[[], None]] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = os.getenv("DEEPGRAM_MODEL", model)
        self.endpointing_ms = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", str(endpointing_ms)))

        self.on_interim = on_interim or (lambda text, speaker: None)
        self.on_final = on_final or (lambda text, speaker, conf: None)
        self.on_eot = on_eot or (lambda: None)

        self.client = AsyncDeepgramClient(api_key=self.api_key)
        self.connection = None
        self.is_connected = False
        self._last_interim_text = ""
        self._last_final_text = ""
        self._last_eot_at = 0.0

    async def connect_and_stream(self, audio_stream) -> None:
        """Connect to Deepgram Flux and stream 16 kHz mono PCM audio chunks."""
        logger.info("Connecting to Deepgram Flux (%s)...", self.model)

        try:
            eot_timeout_ms = max(self.endpointing_ms, 500)
            if eot_timeout_ms != self.endpointing_ms:
                logger.warning(
                    "Deepgram Flux rejected eot_timeout_ms below 500; using %sms",
                    eot_timeout_ms,
                )

            async with self.client.listen.v2.connect(
                model=self.model,
                encoding="linear16",
                sample_rate=16000,
                eager_eot_threshold=0.3,
                eot_threshold=0.5,
                eot_timeout_ms=eot_timeout_ms,
            ) as connection:
                self.connection = connection
                self.is_connected = True
                logger.info("Connected to Deepgram")

                listen_task = asyncio.create_task(self._listen_for_turns(connection))

                try:
                    async for audio_chunk in audio_stream:
                        if not self.is_connected:
                            break
                        await connection.send_media(audio_chunk)
                except asyncio.CancelledError:
                    logger.info("Deepgram media stream cancelled")
                    raise
                finally:
                    self.is_connected = False
                    await connection.send_close_stream()
                    listen_task.cancel()
                    try:
                        await listen_task
                    except asyncio.CancelledError:
                        pass

        except asyncio.CancelledError:
            raise
        except ApiError as exc:
            status_code = getattr(exc, "status_code", "unknown")
            body = str(getattr(exc, "body", ""))[:200]
            logger.error("Deepgram API error (status %s): %s", status_code, body)
            raise RuntimeError(f"Deepgram API error {status_code}: {body}") from None
        except Exception:
            logger.exception("Deepgram connection error")
            raise
        finally:
            self.is_connected = False
            self.connection = None
            logger.info("Deepgram stream finished")

    async def _listen_for_turns(self, connection) -> None:
        async for message in connection:
            message_type = self._get_field(message, "type", "")
            if message_type == "Connected":
                logger.info("Deepgram connection opened")
                continue
            if message_type == "TurnInfo":
                self._handle_turn_info(message)
                continue
            if message_type in {"FatalError", "ConfigureFailure"}:
                logger.error("Deepgram %s: %s", message_type, message)
                self.is_connected = False
                break
            logger.debug("Deepgram message: %s", message)

    @staticmethod
    def _get_field(source, key: str, default=None):
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _handle_turn_info(self, turn_info) -> None:
        text = (self._get_field(turn_info, "transcript", "") or "").strip()
        if not text:
            return

        event = self._get_field(turn_info, "event", "")
        confidence = float(self._get_field(turn_info, "end_of_turn_confidence", 1.0) or 1.0)
        speaker = self._extract_speaker(turn_info)

        if event in {"Update", "StartOfTurn", "EagerEndOfTurn"}:
            if text != self._last_interim_text:
                self._last_interim_text = text
                logger.info("[INTERIM] %s", text)
                self.on_interim(text, speaker)
            return

        if event == "EndOfTurn":
            if text != self._last_final_text:
                self._last_final_text = text
                logger.info("[FINAL] %s (conf=%.2f)", text, confidence)
                self.on_final(text, speaker, confidence)
            self._emit_eot("EndOfTurn")

    def _emit_eot(self, source: str) -> None:
        now = time.perf_counter()
        if now - self._last_eot_at < 0.25:
            return

        self._last_eot_at = now
        logger.info("End-of-turn detected (%s)", source)
        self.on_eot()

    def _extract_speaker(self, turn_info) -> int:
        speaker = self._get_field(turn_info, "speaker", None)
        if speaker is not None:
            return int(speaker)

        words = self._get_field(turn_info, "words", []) or []
        for word in words:
            word_speaker = self._get_field(word, "speaker", None)
            if word_speaker is not None:
                return int(word_speaker)

        return 0

    def disconnect(self) -> None:
        self.is_connected = False


async def audio_generator_from_microphone(sample_rate: int = 16000, chunk_size: int = 2048):
    """Yield microphone audio chunks as raw 16-bit PCM bytes."""
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice is not installed")
        raise

    input_device = os.getenv("AUDIO_INPUT_DEVICE")
    device = int(input_device) if input_device not in {None, ""} else None
    device_info = sd.query_devices(device, "input")
    capture_rate = int(float(device_info.get("default_samplerate", sample_rate)))
    capture_channels = 1
    capture_chunk_size = max(256, int(chunk_size * capture_rate / sample_rate))

    logger.info(
        "Starting microphone capture: device=%s rate=%sHz -> %sHz",
        device_info.get("name", "default"),
        capture_rate,
        sample_rate,
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
    rate_state = None
    last_level_log = 0.0
    emitted_chunks = 0

    def enqueue(data: bytes) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(data)

    def callback(indata, frames, time_info, status):
        if status:
            logger.warning("Microphone status: %s", status)
        loop.call_soon_threadsafe(enqueue, bytes(indata))

    stream = sd.RawInputStream(
        device=device,
        channels=capture_channels,
        samplerate=capture_rate,
        blocksize=capture_chunk_size,
        dtype="int16",
        callback=callback,
    )

    with stream:
        while True:
            audio_chunk = await queue.get()
            if capture_rate != sample_rate:
                audio_chunk, rate_state = audioop.ratecv(
                    audio_chunk,
                    2,
                    capture_channels,
                    capture_rate,
                    sample_rate,
                    rate_state,
                )

            emitted_chunks += 1
            now = time.perf_counter()
            if now - last_level_log >= 2.0:
                rms = audioop.rms(audio_chunk, 2) if audio_chunk else 0
                logger.info("Microphone level rms=%s chunks=%s", rms, emitted_chunks)
                last_level_log = now

            yield audio_chunk


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        def on_interim(text: str, speaker: int):
            print(f"[{speaker}] interim: {text}")

        def on_final(text: str, speaker: int, confidence: float):
            print(f"[{speaker}] final ({confidence:.2f}): {text}")

        def on_eot():
            print("END OF TURN")

        client = DeepgramSTTClient(
            on_interim=on_interim,
            on_final=on_final,
            on_eot=on_eot,
        )

        try:
            await client.connect_and_stream(audio_generator_from_microphone())
        except KeyboardInterrupt:
            logger.info("Interrupted")

    asyncio.run(main())
