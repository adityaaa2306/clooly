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
        "You are a real-time interview copilot. "
        "ALWAYS produce a visible answer immediately. NEVER respond with only reasoning. "
        "Even for casual or unclear questions, give 2-3 bullet points. "
        "Max 120 words. No preamble. Start your response immediately."
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
        configured_max_tokens = int(os.getenv("MAX_TOKENS", max_tokens))
        max_tokens_cap = int(os.getenv("NIM_MAX_TOKENS_CAP", "120"))
        self.max_tokens = min(configured_max_tokens, max_tokens_cap)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=float(os.getenv("NIM_HTTP_TIMEOUT_SECONDS", "4.0")),
        )

        logger.info(
            "NIM client initialized: model=%s temp=%.2f top_p=%.2f max_tokens=%s",
            self.model,
            self.temperature,
            self.top_p,
            self.max_tokens,
        )

    async def stream_answer(
        self, question: str, context: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream only visible content tokens from NIM."""
        if context:
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            user_prompt = f"Question: {question}\nAnswer:"

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
                    max_tokens=self.max_tokens,
                    stream=True,
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
                    fallback_answer = self._create_non_streaming_answer(user_prompt)
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

    def _create_non_streaming_answer(self, user_prompt: str) -> str:
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
                max_tokens=self.max_tokens,
                stream=False,
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
