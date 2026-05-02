"""End-to-end latency benchmark."""

import asyncio
import math
import sys
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.context import ContextEngine
from backend.pipeline import CopilotPipeline


@pytest.fixture
def test_audio_file(tmp_path):
    """Generate a synthetic mono 16 kHz WAV file."""
    sample_rate = 16000
    duration_seconds = 5
    frequency = 440
    audio_file = tmp_path / "test_audio.wav"

    with wave.open(str(audio_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(sample_rate * duration_seconds):
            sample = math.sin(2 * math.pi * frequency * (i / sample_rate))
            value = int(sample * 32767 * 0.3)
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))

        wf.writeframes(bytes(frames))

    return str(audio_file)


async def audio_stream_from_file(audio_file: str, chunk_size: int = 2048):
    """Yield raw audio chunks from a WAV file."""
    with wave.open(audio_file, "rb") as wf:
        while True:
            frame = wf.readframes(chunk_size)
            if not frame:
                break
            yield frame
            await asyncio.sleep(chunk_size / 16000 * 0.5)


class MockSTTClient:
    """Mock STT client for testing without a Deepgram API call."""

    def __init__(self, on_final=None, on_eot=None):
        self.on_final = on_final or (lambda text, speaker, confidence: None)
        self.on_eot = on_eot or (lambda: None)

    async def connect_and_stream(self, audio_stream):
        chunk_count = 0
        max_chunks = 20

        async for _chunk in audio_stream:
            chunk_count += 1
            await asyncio.sleep(0.05)

            if chunk_count >= max_chunks:
                self.on_final(
                    "What is a binary search tree?",
                    speaker=0,
                    confidence=0.95,
                )
                self.on_eot()
                break

    def disconnect(self):
        return None


class MockNIMClient:
    """Mock NIM client for testing without a real API call."""

    def __init__(self):
        self.questions = []

    async def stream_answer(self, question: str, context: str = ""):
        self.questions.append(question)
        await asyncio.sleep(0.3)

        tokens = [
            "A ",
            "binary ",
            "search ",
            "tree ",
            "is ",
            "a ",
            "sorted ",
            "binary ",
            "tree ",
            "where ",
            "left ",
            "child ",
            "< ",
            "parent ",
            "< ",
            "right ",
            "child. ",
            "It ",
            "enables ",
            "O(log ",
            "n) ",
            "search.",
        ]

        for token in tokens:
            yield token
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_latency_e2e(test_audio_file):
    """Measure time from EOT to first token and complete streamed answer."""
    context = ContextEngine()
    mock_nim = MockNIMClient()

    pipeline = CopilotPipeline(
        llm_client=mock_nim,
        context_engine=context,
        stt_client=None,
    )

    timing_data = {
        "eot_time": None,
        "first_token_time": None,
        "stream_complete_time": None,
    }

    original_on_message = pipeline.on_message

    async def track_timing(msg: dict):
        if msg.get("type") == "status" and msg.get("state") == "processing":
            timing_data["eot_time"] = time.perf_counter()
        elif msg.get("type") == "answer" and msg.get("chunk"):
            if timing_data["first_token_time"] is None:
                timing_data["first_token_time"] = time.perf_counter()
        elif msg.get("type") == "answer" and not msg.get("chunk"):
            timing_data["stream_complete_time"] = time.perf_counter()

        await original_on_message(msg)

    pipeline.on_message = track_timing

    mock_stt = MockSTTClient(
        on_final=pipeline._on_final,
        on_eot=pipeline._on_eot_handler,
    )
    pipeline.stt = mock_stt

    try:
        await asyncio.wait_for(
            pipeline.run_with_audio_stream(audio_stream_from_file(test_audio_file)),
            timeout=30,
        )
    except asyncio.TimeoutError:
        pytest.fail("Pipeline timeout")

    await asyncio.sleep(2)

    eot_time = timing_data["eot_time"]
    first_token_time = timing_data["first_token_time"]
    complete_time = timing_data["stream_complete_time"]

    assert eot_time is not None, "EOT time not recorded"
    assert first_token_time is not None, "First token time not recorded"
    assert complete_time is not None, "Stream complete time not recorded"

    time_to_first_token_ms = (first_token_time - eot_time) * 1000
    total_time_ms = (complete_time - eot_time) * 1000

    print("\nLatency Test Results:")
    print(f"   EOT detected at: {eot_time}")
    print(f"   First token at: {first_token_time} ({time_to_first_token_ms:.1f}ms from EOT)")
    print(f"   Stream complete at: {complete_time} ({total_time_ms:.1f}ms from EOT)")

    logs = pipeline.get_timing_logs()
    print("\nPipeline Timing Logs:")
    for key, value in logs.items():
        print(f"   {key}: {value}")

    print("\nASSERTION 1: Time to First Token (TTFT)")
    assert time_to_first_token_ms < 1100, (
        f"TTFT {time_to_first_token_ms:.1f}ms exceeds 1100ms SLA"
    )
    print(f"   PASS {time_to_first_token_ms:.1f}ms < 1100ms")

    print("\nASSERTION 2: Total Time to Answer (TTA)")
    assert total_time_ms < 1500, (
        f"TTA {total_time_ms:.1f}ms exceeds 1500ms SLA"
    )
    print(f"   PASS {total_time_ms:.1f}ms < 1500ms")

    print("\nLATENCY TEST PASSED - All assertions green")


class FastMockNIMClient:
    def __init__(self):
        self.questions = []

    async def stream_answer(self, question: str, context: str = ""):
        self.questions.append(question)
        yield "ok"


@pytest.mark.asyncio
async def test_eot_does_not_reuse_stale_question():
    llm = FastMockNIMClient()
    pipeline = CopilotPipeline(llm_client=llm, context_engine=ContextEngine())
    messages = []

    async def on_message(message):
        messages.append(message)

    pipeline.on_message = on_message

    pipeline._on_final("How far is Delhi from Pune?", speaker=0, confidence=0.9)
    pipeline._on_eot_handler()
    await pipeline.pending_answer_task

    pipeline._on_final("situation", speaker=0, confidence=0.5)
    pipeline._on_eot_handler()
    await pipeline.pending_answer_task

    assert llm.questions == ["How far is Delhi from Pune?"]


@pytest.mark.asyncio
async def test_obvious_question_without_punctuation_triggers_current_turn():
    llm = FastMockNIMClient()
    pipeline = CopilotPipeline(llm_client=llm, context_engine=ContextEngine())
    pipeline.on_message = lambda message: asyncio.sleep(0)

    pipeline._on_final("What is a deadlock", speaker=0, confidence=0.58)
    pipeline._on_eot_handler()
    await pipeline.pending_answer_task

    assert llm.questions == ["What is a deadlock?"]


@pytest.mark.asyncio
async def test_command_shaped_question_triggers_current_turn():
    llm = FastMockNIMClient()
    pipeline = CopilotPipeline(llm_client=llm, context_engine=ContextEngine())
    pipeline.on_message = lambda message: asyncio.sleep(0)

    pipeline._on_final(
        "Please tell me what is the deadlock situation and how would you handle",
        speaker=0,
        confidence=0.60,
    )
    pipeline._on_eot_handler()
    await pipeline.pending_answer_task

    assert llm.questions == [
        "Please tell me what is the deadlock situation and how would you handle?"
    ]


class SlowMockNIMClient:
    def __init__(self):
        self.questions = []

    async def stream_answer(self, question: str, context: str = ""):
        self.questions.append(question)
        await asyncio.sleep(0.08)
        yield "ok"
        await asyncio.sleep(0.08)


@pytest.mark.asyncio
async def test_eot_during_answer_queues_latest_question():
    llm = SlowMockNIMClient()
    pipeline = CopilotPipeline(llm_client=llm, context_engine=ContextEngine())
    pipeline.on_message = lambda message: asyncio.sleep(0)

    pipeline._on_final("What is two plus two?", speaker=0, confidence=0.95)
    pipeline._on_eot_handler()
    first_task = pipeline.pending_answer_task
    await asyncio.sleep(0.02)

    pipeline._on_final(
        "Please tell me what is a deadlock situation and how would you handle",
        speaker=0,
        confidence=0.60,
    )
    pipeline._on_eot_handler()

    await first_task
    if pipeline.pending_answer_task is not first_task:
        await pipeline.pending_answer_task

    assert llm.questions == [
        "What is two plus two?",
        "Please tell me what is a deadlock situation and how would you handle?",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
