"""
Shared Ollama Embedding Client

Factory function for creating LightRAG-compatible embedding functions
using Ollama's API. Used by both single-pipeline (builder.py) and
parallel pipeline (parallel.py) modes.
"""

import asyncio
import logging
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


def create_ollama_embedding_func(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    embedding_dim: int = 768,
    max_token_size: int = 400,
    timeout: float = 120.0,
    max_retries: int = 3,
    retry_base_delay: float = 0.5,
    inter_request_delay: float = 0.0,
):
    """
    Create a LightRAG-compatible async embedding function using Ollama.

    Returns an async function: async (texts: List[str]) -> np.ndarray

    Args:
        ollama_url: Ollama server URL
        embedding_model: Model name for embeddings
        embedding_dim: Expected embedding dimension
        max_token_size: Max text length (chars) to send
        timeout: HTTP request timeout in seconds
        max_retries: Number of retry attempts per text
        retry_base_delay: Base delay between retries (multiplied by attempt)
        inter_request_delay: Delay between individual embedding requests (seconds)
    """
    _client: Optional[httpx.AsyncClient] = None

    async def embedding_func(texts: List[str]) -> np.ndarray:
        nonlocal _client
        if _client is None:
            _client = httpx.AsyncClient(timeout=timeout)

        embeddings = []

        for text in texts:
            if len(text) > max_token_size:
                text = text[:max_token_size]
            text = text.replace('\x00', ' ').replace('\ufffd', ' ')
            if not text.strip():
                text = "empty"

            embedding = None
            for attempt in range(max_retries):
                try:
                    resp = await _client.post(
                        f"{ollama_url}/api/embeddings",
                        json={
                            "model": embedding_model,
                            "prompt": text,
                            "options": {"num_ctx": 2048},
                        },
                    )
                    resp.raise_for_status()
                    embedding = resp.json()["embedding"]
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_base_delay * (attempt + 1))
                    else:
                        logger.error(f"Embedding failed after {max_retries} attempts: {e}")

            embeddings.append(embedding if embedding else [0.0] * embedding_dim)

            if inter_request_delay > 0:
                await asyncio.sleep(inter_request_delay)

        return np.array(embeddings)

    return embedding_func
