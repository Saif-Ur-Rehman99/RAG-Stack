import concurrent.futures
import logging
import time
from typing import Generator

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


class TRTLLMClient:
    """
    OpenAI-compatible client for the TensorRT-LLM server.

    Works against any OpenAI-compatible backend (vLLM, TRT-LLM, etc.)
    so swapping the serving backend never requires changing calling code.
    """

    def __init__(self) -> None:
        self._model = settings.TRTLLM_MODEL
        self._base_url = f"http://localhost:{settings.TRTLLM_HOST_PORT}/v1"
        self._client = OpenAI(base_url=self._base_url, api_key="not-needed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if the server is up and serving the model."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Blocking single-turn completion. Returns the assistant text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Streaming completion — yields text deltas as they arrive.
        Measures and logs TTFT (time to first token) for UX monitoring.
        """
        start = time.time()
        first_token_time: float | None = None

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_time is None:
                    first_token_time = time.time() - start
                    logger.debug("TTFT: %.3fs", first_token_time)
                yield delta

    def benchmark(
        self,
        prompts: list[str],
        max_tokens: int = 64,
        concurrency: int = 16,
    ) -> dict:
        """
        Fire requests in parallel and report aggregate throughput stats.
        This exercises TRT-LLM's in-flight batching — throughput scales
        near-linearly with concurrency until the GPU saturates.

        Returns a dict with wall_time_s, throughput_tok_s, avg_latency_s, etc.
        """
        def _one(prompt: str) -> tuple[int, float]:
            t0 = time.time()
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.usage.completion_tokens, time.time() - t0

        logger.info(
            "Benchmark: %d prompts, concurrency=%d, max_tokens=%d",
            len(prompts), concurrency, max_tokens,
        )
        wall_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(_one, prompts))

        wall = time.time() - wall_start
        total_tokens = sum(tokens for tokens, _ in results)
        avg_latency = sum(lat for _, lat in results) / len(results)

        stats = {
            "num_requests": len(prompts),
            "wall_time_s": round(wall, 2),
            "total_output_tokens": total_tokens,
            "throughput_tok_s": round(total_tokens / wall, 1),
            "avg_latency_s": round(avg_latency, 2),
        }
        logger.info("Benchmark results: %s", stats)
        return stats
