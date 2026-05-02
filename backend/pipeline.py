"""Async orchestrator: STT -> Context -> LLM -> WebSocket.

Architectural Layers:
1. Transcription: DeepgramSTTClient (real-time streaming speech-to-text)
2. Question Detection: ContextEngine (5-stage pipeline with semantic classifier)
3. Utterance Classification: UtteranceClassifier (8-type routing)
4. Dialogue State: DialogueStateManager (tracks topic, misconceptions, confidence)
5. Statement Assessment: StatementAssessor (evaluates correctness, finds misconceptions)
6. Pedagogical Strategy: PedagogicalStrategyEngine (selects response type)
7. Advanced Robustness:
   - SilenceDetector: Detects 5s+ no-response confusion signals
   - RepetitionTracker: Avoids repeating explanations
   - CoreferenceResolver: Resolves pronouns to actual concepts
"""

import asyncio
import logging
import os
import time
from typing import Any, Callable, Coroutine, Optional

from backend.context import ContextEngine
from backend.coreference_resolver import CoreferenceResolver
from backend.dialogue_state import DialogueState, DialogueStateManager
from backend.llm import NIMClient
from backend.repetition_tracker import RepetitionTracker
from backend.response_strategy import PedagogicalStrategyEngine, ResponseStrategy
from backend.silence_detector import SilenceDetector
from backend.stt import DeepgramSTTClient
from backend.statement_assessment import StatementAssessor
from backend.utterance_classifier import UtteranceClassifier

logger = logging.getLogger(__name__)


class CopilotPipeline:
    """Orchestrates STT, context management, LLM streaming, and UI messages.

    This pipeline implements the complete Cluely learning copilot architecture:
    - STT → question detection → utterance classification
    - Dialogue state tracking across exchanges
    - Statement assessment and misconception identification
    - Pedagogical strategy selection
    - Advanced robustness: silence detection, repetition avoidance, coreference resolution
    """

    def __init__(
        self,
        llm_client: Optional[NIMClient] = None,
        context_engine: Optional[ContextEngine] = None,
        stt_client: Optional[DeepgramSTTClient] = None,
        on_message: Optional[Callable[[dict], Coroutine[Any, Any, None]]] = None,
    ):
        # Core components
        self.llm = llm_client or NIMClient()
        self.context = context_engine or ContextEngine()
        self.stt = stt_client

        async def default_callback(msg: dict) -> None:
            return None

        self.on_message = on_message or default_callback
        
        # Processing state
        self.is_processing_answer = False
        self.timing_logs: dict[str, float] = {}
        self.pending_answer_task: Optional[asyncio.Task] = None
        
        # Speaker tracking
        self._speaker_roles: dict[int, str] = {}
        
        # Utterance sequencing
        self._final_sequence = 0
        self._last_final_record: Optional[dict] = None
        self._last_processed_final_id = 0
        self._queued_final_records: list[dict] = []
        
        # ============ ARCHITECTURAL LAYER INITIALIZATION ============
        
        # Layer 3: Utterance Classification (8-type routing)
        self.utterance_classifier = UtteranceClassifier()
        
        # Layer 4: Dialogue State Management
        self.dialogue_state_manager = DialogueStateManager()
        
        # Layer 5: Statement Assessment (correctness + misconceptions)
        self.statement_assessor = StatementAssessor()
        
        # Layer 6: Pedagogical Strategy Selection
        self.strategy_engine = PedagogicalStrategyEngine()
        
        # Layer 7: Advanced Robustness
        self.silence_detector = SilenceDetector()
        self.repetition_tracker = RepetitionTracker()
        self.coreference_resolver = CoreferenceResolver()
        
        logger.info("Pipeline initialized with full 7-layer architecture")


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
        """Process a final transcript.
        
        Feeds data into architectural layers:
        - Dialogue state tracking
        - Utterance classification
        - Silence detector reset (response received)
        - Coreference resolver context update
        """
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

        # ============ LAYER 3-4: Classify utterance and update dialogue state ============
        try:
            # Classify utterance type
            utterance_classification = self.utterance_classifier.classify(text)
            logger.info("Utterance classified: %s", utterance_classification.type.value)
            
            # Update dialogue state based on classification
            if speaker_name == "user":
                utterance_type = utterance_classification.type.value
                # Track user confidence signals
                if utterance_type == "correct_statement":
                    self.dialogue_state_manager.state.record_correct_response()
                elif utterance_type == "incorrect_statement":
                    self.dialogue_state_manager.state.record_incorrect_response()
                elif utterance_type == "frustrated_statement":
                    self.dialogue_state_manager.state.record_confused_response()
                elif utterance_type in ["clarification_question", "confirmation_question"]:
                    self.dialogue_state_manager.state.record_clarification_request()
            
            # Extract concepts from utterance for coreference tracking
            concepts = self.coreference_resolver.extract_noun_phrases(text)
            self.coreference_resolver.update_context(text, concepts)
            
            # Signal silence detector that we received a response
            if speaker_name == "user":
                self.silence_detector.record_utterance(
                    utterance_classification.type.value,
                    is_question=utterance_classification.type.value in [
                        "direct_question", "clarification_question", "confirmation_question"
                    ]
                )
            
        except Exception as e:
            logger.error("Error processing final transcript: %s", e, exc_info=True)
            # Don't fail the pipeline on classification errors; proceed anyway

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
        """Handle end-of-turn detection from STT.
        
        Also checks for silence signals (prolonged no-response after question).
        """
        if self.pending_answer_task and not self.pending_answer_task.done():
            if self._last_final_record:
                self._queued_final_records.append(dict(self._last_final_record))
                self._queued_final_records = self._queued_final_records[-6:]
            logger.info("EOT received while answer task is still running; queued latest turn")
            return

        logger.info("EOT detected")
        
        # ============ LAYER 7: Check for silence signals ============
        try:
            silence_event = self.silence_detector.check_silence()
            if silence_event:
                logger.warning(f"Silence signal detected: {silence_event}")
                # This can be used for adaptive interventions in future
        except Exception as e:
            logger.error("Error checking silence: %s", e)
        
        self.pending_answer_task = asyncio.create_task(self._process_answer_on_eot())


    async def _process_answer_on_eot(self, queued_record: Optional[dict] = None) -> None:
        """Start LLM streaming after STT reports end-of-turn.
        
        Full architectural pipeline:
        1. Extract question and context
        2. Check for repetition (avoid explaining same thing twice)
        3. Resolve coreferences (pronouns → concepts)
        4. Select pedagogical strategy based on dialogue state
        5. Stream LLM response with strategy-guided system prompt
        6. Track explanation for future repetition avoidance
        """
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

            # ============ LAYER 5-6: Check repetition and select strategy ============
            
            # Check if this exact explanation was recently given (repetition avoidance)
            is_repetition = False
            try:
                is_repetition, original_record = self.repetition_tracker.check_repetition(
                    question, 
                    self.dialogue_state_manager.state.topic or ""
                )
                if is_repetition:
                    logger.info(
                        f"Question within recent topic coverage; requesting variation "
                        f"(original asked {original_record.age_seconds():.0f}s ago)"
                    )
            except Exception as e:
                logger.error("Error checking repetition: %s", e)

            # Resolve coreferences in the question (pronouns → concepts)
            try:
                resolved_question = self.coreference_resolver.resolve_in_sentence(
                    question,
                    topic=self.dialogue_state_manager.state.topic
                )
                if resolved_question != question:
                    logger.info(f"Resolved coreferences: '{question}' → '{resolved_question}'")
                    question = resolved_question
            except Exception as e:
                logger.error("Error resolving coreferences: %s", e)

            # Select pedagogical strategy
            strategy: Optional[ResponseStrategy] = None
            try:
                # Classify the question utterance
                utterance_type = self.utterance_classifier.classify(question).type
                
                # Generate pedagogical instruction
                strategy = self.strategy_engine.choose_strategy(
                    utterance_type=utterance_type,
                    dialogue_state=self.dialogue_state_manager.state
                )
                logger.info(f"Selected pedagogical strategy: {strategy}")
            except Exception as e:
                logger.error("Error selecting strategy: %s", e)
                # Fallback to neutral strategy
                strategy = self.strategy_engine.strategies.get("explain")

            # Build enhanced system prompt with pedagogical guidance
            enhanced_system_prompt = None
            if strategy and self.dialogue_state_manager.state:
                try:
                    enhanced_system_prompt = self.strategy_engine.generate_system_prompt_instruction(
                        self.dialogue_state_manager.state,
                        strategy
                    )
                except Exception as e:
                    logger.error("Error generating enhanced prompt: %s", e)

            first_token_time = None
            answer_tokens: list[str] = []
            llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "2.5"))

            try:
                async with asyncio.timeout(llm_timeout):
                    async for token in self.llm.stream_answer(
                        question,
                        context_summary,
                        system_prompt_override=enhanced_system_prompt
                    ):
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
            except TypeError as e:
                # Handle case where llm.stream_answer doesn't accept system_prompt_override
                logger.warning(
                    "LLM client doesn't support system_prompt_override, retrying without: %s", e
                )
                try:
                    async with asyncio.timeout(llm_timeout):
                        async for token in self.llm.stream_answer(question, context_summary):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                time_to_first_token = (first_token_time - eot_time) * 1000
                                self.timing_logs["first_token_received"] = first_token_time
                                self.timing_logs["time_to_first_token_ms"] = time_to_first_token

                            await self.on_message({
                                "type": "answer",
                                "text": token,
                                "chunk": True,
                            })
                            answer_tokens.append(token)
                except Exception:
                    logger.exception("Fallback LLM stream error")
                    await self.on_message({"type": "status", "state": "error"})
                    return
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
                
                # ============ LAYER 7: Track explanation for repetition avoidance ============
                try:
                    self.repetition_tracker.add_explanation(
                        topic=self.dialogue_state_manager.state.topic or "general",
                        explanation=full_answer,
                        concept_covered=question[:50],  # First 50 chars of question as concept ID
                        depth_level="intermediate"
                    )
                except Exception as e:
                    logger.error("Error tracking explanation: %s", e)
                
                # ============ LAYER 4: Update dialogue state with response ============
                try:
                    self.dialogue_state_manager.state.last_response_type = strategy.type if strategy else "unknown"
                    self.dialogue_state_manager.state.add_explanation(
                        topic=self.dialogue_state_manager.state.topic or "general",
                        explanation=full_answer
                    )
                except Exception as e:
                    logger.error("Error updating dialogue state: %s", e)
                
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
