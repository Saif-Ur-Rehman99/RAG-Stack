import logging
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


class VLLMEmbeddingClient:
    """
    Drop-in replacement for EmbeddingModelSingleton backed by a running
    vLLM server in embedding mode (--task embed).

    Matches EmbeddingModelSingleton's interface exactly:
      - model_id            → str
      - embedding_size      → int
      - max_input_length    → int
      - __call__(texts, to_list) → embeddings

    Enable by setting USE_VLLM_EMBEDDING=true in .env and starting the server:
        python3 start_embedding_server.py
    """

    def __init__(self) -> None:
        self._model_id = settings.TEXT_EMBEDDING_MODEL_ID
        self._base_url = f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1"
        self._client = OpenAI(base_url=self._base_url, api_key="not-needed")
        self._assert_healthy()

    # ------------------------------------------------------------------
    # EmbeddingModelSingleton-compatible interface
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @cached_property
    def embedding_size(self) -> int:
        probe = self._embed(["probe"])
        return len(probe[0])

    @cached_property
    def max_input_length(self) -> int:
        try:
            models = self._client.models.list()
            for m in models.data:
                if hasattr(m, "max_model_len") and m.max_model_len:
                    return m.max_model_len
        except Exception:
            pass
        return 512   # safe default for all-MiniLM-L6-v2

    def __call__(
        self,
        input_text: str | list[str],
        to_list: bool = True,
    ) -> NDArray[np.float32] | list[float] | list[list[float]]:
        if isinstance(input_text, str):
            input_text = [input_text]

        try:
            embeddings = self._embed(input_text)
        except Exception as exc:
            logger.error("vLLM embedding request failed: %s", exc)
            return [] if to_list else np.array([])

        if to_list:
            return embeddings
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_healthy(self) -> None:
        try:
            self._client.models.list()
            logger.info("vLLM server healthy at %s (model: %s)", self._base_url, self._model_id)
        except Exception:
            raise RuntimeError(
                f"Cannot reach vLLM server at {self._base_url}.\n"
                "Start it first:  python3 start_embedding_server.py"
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model_id,
            input=texts,
        )
        response.data.sort(key=lambda x: x.index)
        return [item.embedding for item in response.data]
