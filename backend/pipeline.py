"""Async orchestrator: STT -> Context -> LLM -> WebSocket."""

import asyncio
import logging
import os
import time
from typing import Any, Callable, Coroutine, Optional

from backend.context import ContextEngine
from backend.llm import NIMClient
from backend.stt import DeepgramSTTClient

logger = logging.getLogger(__name__)


class CopilotPipeline:
    """Orchestrates STT, context management, LLM streaming, and UI messages."""

    def __init__(
        self,
        llm_client: Optional[NIMClient] = None,
        context_engine: Optional[ContextEngine] = None,
        stt_client: Optional[DeepgramSTTClient] = None,
        on_message: Optional[Callable[[dict], Coroutine[Any, Any, None]]] = None,
    ):
        self.llm = llm_client or NIMClient()
        self.context = context_engine or ContextEngine()
        self.stt = stt_client

        async def default_callback(msg: dict) -> None:
            return None

        self.on_message = on_message or default_callback
        self.is_processing_answer = False
        self.timing_logs: dict[str, float] = {}
        self.pending_answer_task: Optional[asyncio.Task] = None
        self._speaker_roles: dict[int, str] = {}
        self._final_sequence = 0
        self._last_final_record: Optional[dict] = None
        self._last_processed_final_id = 0
        self._queued_final_records: list[dict] = []

    async def initialize_stt(self) -> None:
        """Initialize the STT client with pipeline callbacks."""
        if self.stt is None:
            self.stt = DeepgramSTTClient(
                on_interim=self._on_interim,
                on_final=self._on_final,
                on_eot=self._on_eot_handler,
            )

    def _on_interim(self, text: str, speaker: int) -> None:
        speaker_name = self._speaker_name(speaker)
        logger.debug("[INTERIM] %s: %s", speaker_name, text)
        asyncio.create_task(
            self.on_message(
                {
                    "type": "transcript",
                    "text": text,
                    "speaker": speaker_name,
                    "is_final": False,
                }
            )
        )

    def _on_final(self, text: str, speaker: int, confidence: float) -> None:
        speaker_name = self._speaker_name(speaker)
        logger.info("[FINAL] %s (conf=%.2f): %s", speaker_name, confidence, text)

        context_record = self.context.add_transcript(
            speaker_name,
            text,
            is_final=True,
            confidence=confidence,
        )
        self._final_sequence += 1
        self._last_final_record = {
            **(context_record or {}),
            "id": self._final_sequence,
        }

        asyncio.create_task(
            self.on_message(
                {
                    "type": "transcript",
                    "text": self._last_final_record.get("text", text),
                    "speaker": speaker_name,
                    "is_final": True,
                }
            )
        )

    def _speaker_name(self, speaker: int) -> str:
        """First detected speaker is interviewer; the other speaker is candidate/user."""
        if speaker not in self._speaker_roles:
            self._speaker_roles[speaker] = "interviewer" if not self._speaker_roles else "user"
        return self._speaker_roles[speaker]

    def _on_eot_handler(self) -> None:
        """Handle end-of-turn detection from STT."""
        if self.pending_answer_task and not self.pending_answer_task.done():
            if self._last_final_record:
                self._queued_final_records.append(dict(self._last_final_record))
                self._queued_final_records = self._queued_final_records[-6:]
            logger.info("EOT received while answer task is still running; queued latest turn")
            return

        logger.info("EOT detected")
        self.pending_answer_task = asyncio.create_task(self._process_answer_on_eot())

    async def _process_answer_on_eot(self, queued_record: Optional[dict] = None) -> None:
        """Start LLM streaming after STT reports end-of-turn."""
        if self.is_processing_answer:
            logger.warning("Already processing answer, skipping")
            return

        self.is_processing_answer = True
        eot_time = time.perf_counter()
        self.timing_logs = {"eot_detected": eot_time}

        try:
            final_record = queued_record or self._last_final_record
            if not final_record or final_record.get("id") == self._last_processed_final_id:
                logger.warning("No new finalized utterance for this EOT, skipping LLM")
                await self.on_message({"type": "status", "state": "listening"})
                return

            self._last_processed_final_id = final_record["id"]
            question = self.context.get_question_from_record(final_record)
            if not question:
                logger.info(
                    "Latest finalized interviewer utterance is not a question, skipping LLM: %s",
                    final_record.get("text", ""),
                )
                await self.on_message({"type": "status", "state": "listening"})
                return

            logger.info("Processing question: %s", question)
            context_summary = self.context.get_summary()
            await self.on_message({"type": "status", "state": "processing"})

            first_token_time = None
            answer_tokens: list[str] = []
            llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "2.5"))

            try:
                async with asyncio.timeout(llm_timeout):
                    async for token in self.llm.stream_answer(question, context_summary):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            time_to_first_token = (first_token_time - eot_time) * 1000
                            self.timing_logs["first_token_received"] = first_token_time
                            self.timing_logs["time_to_first_token_ms"] = time_to_first_token
                            logger.info(
                                "First token received in %.1fms from EOT",
                                time_to_first_token,
                            )

                        await self.on_message(
                            {
                                "type": "answer",
                                "text": token,
                                "chunk": True,
                            }
                        )
                        answer_tokens.append(token)

            except asyncio.TimeoutError:
                logger.error("LLM stream timeout (>%.0fms)", llm_timeout * 1000)
                await self.on_message({"type": "status", "state": "timeout"})
                return
            except asyncio.CancelledError:
                logger.info("LLM stream cancelled")
                raise
            except Exception:
                logger.exception("LLM streaming error")
                await self.on_message({"type": "status", "state": "error"})
                return
            finally:
                complete_time = time.perf_counter()
                total_time = (complete_time - eot_time) * 1000
                self.timing_logs["stream_complete"] = complete_time
                self.timing_logs["total_time_ms"] = total_time

                await self.on_message({"type": "answer", "text": "", "chunk": False})

                full_answer = "".join(answer_tokens)
                logger.info("Answer complete in %.1fms (TTA)", total_time)
                logger.info("Answer preview: %s", full_answer[:100])
                self._log_timing_summary()

        finally:
            self.is_processing_answer = False
            await self.on_message({"type": "status", "state": "listening"})
            queued_question = self._pop_latest_queued_question()
            if queued_question:
                logger.info(
                    "Processing queued question after current answer: %s",
                    queued_question.get("text", ""),
                )
                self.pending_answer_task = asyncio.create_task(
                    self._process_answer_on_eot(queued_question)
                )

    def _pop_latest_queued_question(self) -> Optional[dict]:
        """Return the newest queued finalized interviewer question, if any."""
        if not self._queued_final_records:
            return None

        queued_records = self._queued_final_records
        self._queued_final_records = []
        for record in reversed(queued_records):
            if record.get("id") == self._last_processed_final_id:
                continue
            if self.context.get_question_from_record(record):
                return record
        return None

    def _log_timing_summary(self) -> None:
        if "time_to_first_token_ms" not in self.timing_logs:
            logger.warning("No timing data available")
            return

        ttft = self.timing_logs.get("time_to_first_token_ms", 0)
        total = self.timing_logs.get("total_time_ms", 0)

        logger.info("Timing Summary:")
        logger.info("Time to First Token (TTFT): %.1fms", ttft)
        logger.info("Total Time to Answer (TTA): %.1fms", total)

        if ttft > 1100:
            logger.warning("TTFT exceeded 1100ms SLA: %.1fms", ttft)
        if total > 1500:
            logger.error("TTA exceeded 1500ms SLA: %.1fms", total)

    async def run_with_audio_stream(self, audio_stream) -> None:
        """Run the full pipeline with an async audio byte stream."""
        await self.initialize_stt()

        try:
            await self.on_message({"type": "status", "state": "listening"})
            logger.info("Pipeline started, streaming audio")
            await self.stt.connect_and_stream(audio_stream)

            if self.pending_answer_task:
                try:
                    await asyncio.wait_for(self.pending_answer_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.error("Pending answer task timeout")

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
            raise
        except Exception:
            logger.exception("Pipeline error")
            await self.on_message({"type": "status", "state": "error"})
            raise
        finally:
            if self.stt:
                self.stt.disconnect()
            logger.info("Pipeline stopped")

    def get_timing_logs(self) -> dict:
        """Get timing logs for debugging/testing."""
        return self.timing_logs.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        pipeline = CopilotPipeline()

        async def mock_on_message(msg: dict):
            print(f"{msg['type']}: {str(msg).ljust(80)[:80]}")

        pipeline.on_message = mock_on_message

        async def mock_audio_stream():
            for _ in range(10):
                yield b"mock_audio_bytes" * 100
                await asyncio.sleep(0.5)

        await pipeline.run_with_audio_stream(mock_audio_stream())

    asyncio.run(main())
