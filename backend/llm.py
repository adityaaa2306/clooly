"""NVIDIA NIM streaming LLM client."""

import asyncio
import logging
import os
import threading
from typing import AsyncGenerator

from openai import OpenAI

logger = logging.getLogger(__name__)


class NIMClient:
    """NVIDIA NIM API streaming client using the OpenAI-compatible SDK."""

    SYSTEM_PROMPT = (
        "You are a real-time interview copilot helping a candidate answer questions confidently.\n\n"
        "Your job is NOT to answer the question yourself.\n"
        "Your job is to give the candidate a clear, speakable framework they can say out loud naturally.\n\n"
        "Rules:\n"
        "- Lead with a one-line definition or core concept\n"
        "- Follow with 2-3 key points the candidate should mention\n"
        "- End with one concrete example or analogy if relevant\n"
        "- Answer as completely as the question requires.\n"
        "- If the question is simple, answer in 2-3 bullet points.\n"
        "- If the question asks for multiple concepts or says 'in detail' or 'thoroughly', cover each concept fully before moving on.\n"
        "- Never truncate an answer. Never stop mid-point.\n"
        "- Always finish your last sentence completely.\n"
        "- Use simple language — the candidate reads this in 5-10 seconds and speaks it naturally\n"
        "- Never use markdown headers or bold formatting\n"
        "- Never start with 'Certainly' or 'Sure' or any preamble\n"
        "- If the question is vague, answer the most likely interview interpretation of it"
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 200,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_NIM_API_KEY")
        self.base_url = base_url or os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.model = model or os.getenv("NIM_MODEL", "openai/gpt-oss-20b")
        self.temperature = float(os.getenv("NIM_TEMPERATURE", temperature))
        self.top_p = float(os.getenv("NIM_TOP_P", top_p))
        # Generous default; per-question max_tokens is estimated dynamically by complexity.
        self.default_max_tokens = int(os.getenv("MAX_TOKENS", "1500"))

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=float(os.getenv("NIM_HTTP_TIMEOUT_SECONDS", "4.0")),
        )

        logger.info(
            "NIM client initialized: model=%s temp=%.2f top_p=%.2f default_max_tokens=%s",
            self.model,
            self.temperature,
            self.top_p,
            self.default_max_tokens,
        )

    def _estimate_tokens_for_question(self, question: str) -> int:
        """Estimate required tokens based on question complexity."""
        q = question.lower()
        connectors = ["and", "also", "as well as", "along with", "plus", "both"]
        concept_count = 1 + sum(1 for c in connectors if c in q)

        if any(
            w in q
            for w in [
                "detail",
                "detailed",
                "thoroughly",
                "everything",
                "full",
                "complete",
                "explain",
            ]
        ):
            base = 600
        elif any(w in q for w in ["what is", "define", "meaning"]):
            base = 200
        else:
            base = 350

        return min(base * concept_count, self.default_max_tokens)

    def _is_depth_question(self, question: str) -> bool:
        """Check if the question asks for depth/detailed explanation."""
        depth_keywords = {
            "detailed", "detail", "in depth", "deeply", "thoroughly",
            "comprehensive", "everything", "all about", "cover", "elaborate",
            "expand", "full", "complete", "whole", "entire", "all aspects"
        }
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in depth_keywords)

    async def stream_answer(
        self, question: str, context: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream only visible content tokens from NIM."""
        estimated_max_tokens = self._estimate_tokens_for_question(question)

        # Detect if this is a depth/detailed question
        is_detailed = self._is_depth_question(question)
        
        # Build user prompt with interview framing
        if is_detailed:
            # For detailed questions, emphasize structure and pillars for candidate to expand on
            framework_instruction = "\nFor a DETAILED question: give 3 clear pillars with enough context that the candidate can expand on each one verbally. Don't dump everything, but give enough that they sound informed."
            if context:
                user_prompt = f"Interview question: {question}\nRecent context: {context}{framework_instruction}\n\nGive the candidate a comprehensive framework they can talk through naturally."
            else:
                user_prompt = f"Interview question: {question}{framework_instruction}\n\nGive the candidate a comprehensive framework they can talk through naturally."
        else:
            # For normal questions, concise framing.
            if context:
                user_prompt = f"Interview question: {question}\nRecent context: {context}\n\nGive the candidate a concise framework to answer this out loud. Be clear and natural."
            else:
                user_prompt = f"Interview question: {question}\n\nGive the candidate a concise framework to answer this out loud. Be clear and natural."

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        done = object()

        def push(item: object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def stream_in_thread() -> None:
            token_count = 0
            reasoning_count = 0
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=estimated_max_tokens,
                    stream=True,
                    stop=None,
                    extra_body={"reasoning_effort": "low"}
                )

                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue

                    delta = chunk.choices[0].delta
                    if getattr(delta, "reasoning_content", None):
                        reasoning_count += 1
                        continue

                    content = getattr(delta, "content", None)
                    if content:
                        token_count += 1
                        push(content)

                if token_count == 0:
                    logger.warning(
                        "NIM stream returned no visible content "
                        "(reasoning chunks=%s); retrying once without streaming",
                        reasoning_count,
                    )
                    fallback_answer = self._create_non_streaming_answer(
                        user_prompt, estimated_max_tokens
                    )
                    if fallback_answer:
                        token_count = 1
                        push(fallback_answer)

                logger.info(
                    "NIM stream complete. Total content chunks: %s reasoning chunks skipped: %s",
                    token_count,
                    reasoning_count,
                )
            except Exception as exc:
                push(exc)
            finally:
                push(done)

        threading.Thread(target=stream_in_thread, daemon=True).start()

        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                logger.error("Error streaming answer: %s", item)
                raise item
            yield str(item)

    def _create_non_streaming_answer(
        self, user_prompt: str, max_tokens: int
    ) -> str:
        """Retry once when the stream contains only reasoning chunks."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                top_p=self.top_p,
                max_tokens=max_tokens,
                stream=False,
                stop=None,
            )
        except Exception as exc:
            logger.error("NIM non-streaming retry failed: %s", exc)
            return ""

        if not getattr(completion, "choices", None):
            return ""

        message = completion.choices[0].message
        content = getattr(message, "content", "") or ""
        return content.strip()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def main():
        client = NIMClient()
        async for token in client.stream_answer("What is a binary search tree?", ""):
            print(token, end="", flush=True)
        print()

    asyncio.run(main())
