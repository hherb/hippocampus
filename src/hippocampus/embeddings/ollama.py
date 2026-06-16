"""Ollama-based embedding provider using the ``/api/embed`` endpoint."""

from __future__ import annotations

import logging

import httpx

from hippocampus.config import Settings
from hippocampus.embeddings.base import EmbeddingProvider, TaskType

log = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding provider fails to produce vectors."""


class OllamaEmbedding(EmbeddingProvider):
    """Generate embeddings via a local Ollama instance."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.embedding_model
        self.use_prefix = settings.embedding_prefix
        self.expected_dimensions = settings.embedding_dimensions
        self._client = httpx.AsyncClient(
            timeout=settings.embedding_request_timeout,
        )

    async def embed(
        self, texts: list[str], task_type: TaskType = TaskType.DOCUMENT
    ) -> list[list[float]]:
        """Embed a batch of texts through Ollama.

        Prepends task-type prefixes when ``embedding_prefix`` is enabled
        (required by models like *nomic-embed-text*).

        Raises:
            EmbeddingError: If Ollama is unreachable, returns an error, or
                produces an unexpected response format.
        """
        if not texts:
            return []

        input_texts = texts
        if self.use_prefix:
            input_texts = [f"{task_type.value}: {t}" for t in texts]

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": input_texts},
            )
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"Ollama embedding request timed out ({len(texts)} texts)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        data = resp.json()
        if "embeddings" not in data:
            raise EmbeddingError(
                f"Unexpected Ollama response (missing 'embeddings' key): "
                f"{str(data)[:200]}"
            )

        embeddings = data["embeddings"]
        if len(embeddings) != len(input_texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} embeddings for "
                f"{len(input_texts)} inputs"
            )

        # Guard against a model whose vector size disagrees with the schema's
        # vector(dim) column — otherwise this surfaces as an opaque error deep
        # inside an INSERT. Check the first vector (all are uniform).
        if embeddings:
            got = len(embeddings[0])
            if got != self.expected_dimensions:
                raise EmbeddingError(
                    f"Embedding model '{self.model}' returned {got}-dim vectors, "
                    f"but the schema expects {self.expected_dimensions}. Set "
                    f"HIPPOCAMPUS_EMBEDDING_DIMENSIONS to match the model and "
                    f"re-run 'hippocampus init-db'."
                )

        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
