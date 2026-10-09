"""Batched OpenAI-compatible embeddings through the Portkey gateway."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import ssl
from typing import Any

import httpx
import numpy as np
import truststore
from dotenv import load_dotenv
from openai import AsyncOpenAI, PermissionDeniedError


load_dotenv(dotenv_path=".env", override=True)

from portkey_profiles import model_profile  # noqa: E402


PORTKEY_BASE_URL = os.getenv(
    "PORTKEY_BASE_URL", "https://portkey.bain.dev/v1"
).strip()
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60")
)
LLM_WALL_TIMEOUT_SECONDS = float(
    os.getenv("LLM_WALL_TIMEOUT_SECONDS", "120")
)
DEFAULT_EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))


class EmbeddingError(Exception):
    pass


def get_api_key() -> str:
    return os.getenv("PORTKEY_API_KEY", "").strip()


def get_embedding_model() -> str:
    credential_type = os.getenv("PORTKEY_CREDENTIAL_TYPE")
    return model_profile(credential_type).embedding_model


def _coerce_embedding(value: Any) -> list[float]:
    """Normalize common OpenAI-compatible embedding payload formats."""
    if isinstance(value, dict):
        value = value.get("embedding", value)

    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()

    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]

    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("["):
            parsed = json.loads(raw)
            return [float(item) for item in parsed]

        decoded = base64.b64decode(raw, validate=True)
        if len(decoded) % 4 == 0:
            values = np.frombuffer(decoded, dtype="<f4")
            if values.size:
                return values.astype(float).tolist()

    raise EmbeddingError(
        "Unsupported embedding payload type: "
        f"{type(value).__name__}"
    )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise EmbeddingError("Expected a two-dimensional embedding matrix.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingError("Embedding API returned a zero-length vector.")
    return vectors / norms


class EmbeddingClient:
    """Batched OpenAI embeddings through the configured Portkey gateway."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or get_embedding_model()
        self.batch_size = max(
            1, int(batch_size or DEFAULT_EMBED_BATCH_SIZE)
        )
        self._client = client

    async def _embed_with_client(
        self,
        client: Any,
        texts: list[str],
        normalize: bool,
    ) -> np.ndarray:
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            async with asyncio.timeout(LLM_WALL_TIMEOUT_SECONDS):
                response = client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
                if inspect.isawaitable(response):
                    response = await response

            items = sorted(
                response.data,
                key=lambda item: (
                    item.get("index", 0)
                    if isinstance(item, dict)
                    else getattr(item, "index", 0)
                ),
            )
            for item in items:
                payload = (
                    item.get("embedding")
                    if isinstance(item, dict)
                    else item.embedding
                )
                vectors.append(_coerce_embedding(payload))

        if len(vectors) != len(texts):
            raise EmbeddingError(
                "Embedding API returned an unexpected number of vectors: "
                f"expected {len(texts)}, received {len(vectors)}."
            )

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError(
                "Embedding API returned inconsistent vector dimensions."
            )

        matrix = np.asarray(vectors, dtype=float)
        return _normalize(matrix) if normalize else matrix

    async def embed_texts_async(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError(
                "Every embedding input must be a non-empty string."
            )

        try:
            if self._client is not None:
                return await self._embed_with_client(
                    self._client, texts, normalize
                )

            api_key = get_api_key()
            if not api_key:
                raise EmbeddingError("Portkey API key is not configured.")
            if not PORTKEY_BASE_URL:
                raise EmbeddingError("Portkey base URL is not configured.")

            async with httpx.AsyncClient(
                verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            ) as http_client:
                client = AsyncOpenAI(
                    base_url=PORTKEY_BASE_URL,
                    api_key=api_key,
                    http_client=http_client,
                    max_retries=LLM_MAX_RETRIES,
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                )
                return await self._embed_with_client(
                    client, texts, normalize
                )
        except EmbeddingError:
            raise
        except TimeoutError as exc:
            raise EmbeddingError(
                "Portkey embedding request exceeded the "
                f"{LLM_WALL_TIMEOUT_SECONDS:g}s wall-clock limit."
            ) from exc
        except PermissionDeniedError as exc:
            credential_type = os.getenv(
                "PORTKEY_CREDENTIAL_TYPE", "personal"
            ).strip()
            raise EmbeddingError(
                "Portkey denied access to embedding model "
                f"{self.model!r} for credential type "
                f"{credential_type!r}. Verify that PORTKEY_API_KEY belongs "
                "to the workspace that owns this model alias and has "
                "permission to invoke it."
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Portkey embedding request failed: {exc}"
            ) from exc

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Synchronous compatibility entry point for non-notebook callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed_texts_async(texts, normalize))
        raise RuntimeError(
            "embed_texts() cannot run inside an active event loop; "
            "await embed_texts_async() instead."
        )
