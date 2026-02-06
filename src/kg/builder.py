"""
Knowledge Graph Builder

Core LightRAG builder that integrates with Ollama for entity extraction
and graph building. Configurable for any domain (no hardcoded defaults).
"""

import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Knowledge Graph builder using LightRAG + Ollama.

    Provides the core graph building functionality that can be used
    standalone or composed with the parallel pipeline.
    """

    def __init__(
        self,
        working_dir: str,
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "mistral-small3.2:24b",
        embedding_model: str = "nomic-embed-text",
        embedding_dim: int = 768,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        self.rag = self._init_lightrag()

    def _init_lightrag(self):
        """Initialize LightRAG with Ollama backend."""
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_model_complete
        from lightrag.utils import EmbeddingFunc
        import httpx

        ollama_url = self.ollama_url
        embedding_model = self.embedding_model
        embedding_dim = self.embedding_dim

        logger.info(f"Configuring LightRAG - LLM: {self.llm_model}, Embedding: {embedding_model}")

        async def embedding_func(texts: List[str]) -> np.ndarray:
            """Direct HTTP embedding with robust error handling."""
            total = len(texts)
            logger.info(f"Embedding batch: {total} texts")
            embeddings = []

            async with httpx.AsyncClient(timeout=300.0) as client:
                # Warmup
                logger.info("Warming up embedding model...")
                for _ in range(3):
                    try:
                        warmup = await client.post(
                            f"{ollama_url}/api/embeddings",
                            json={
                                "model": embedding_model,
                                "prompt": "warmup",
                                "options": {"num_ctx": 2048}
                            }
                        )
                        if warmup.status_code == 200:
                            logger.info("Embedding model ready")
                            break
                    except Exception as e:
                        logger.warning(f"Warmup attempt failed: {e}")
                    await asyncio.sleep(5.0)

                await asyncio.sleep(2.0)
                for i, text in enumerate(texts):
                    if len(text) > 400:
                        text = text[:400]
                    text = text.replace('\x00', ' ').replace('\ufffd', ' ')
                    if not text.strip():
                        text = "empty"

                    embedding = None
                    for attempt in range(5):
                        try:
                            resp = await client.post(
                                f"{ollama_url}/api/embeddings",
                                json={
                                    "model": embedding_model,
                                    "prompt": text,
                                    "options": {"num_ctx": 2048}
                                }
                            )
                            resp.raise_for_status()
                            embedding = resp.json()["embedding"]
                            break
                        except Exception as e:
                            wait = 2.0 * (attempt + 1)
                            logger.warning(f"Retry {attempt+1}/5 for text {i}: {str(e)[:50]}...")
                            if attempt < 4:
                                await asyncio.sleep(wait)
                            else:
                                logger.error(f"Failed embedding text {i} after 5 attempts")

                    embeddings.append(embedding if embedding else [0.0] * embedding_dim)

                    if (i + 1) % 50 == 0:
                        logger.info(f"Embedding progress: {i+1}/{total}")

                    await asyncio.sleep(0.5)

            logger.info(f"Completed embedding batch: {total} texts")
            return np.array(embeddings)

        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            llm_model_kwargs={
                "host": self.ollama_url,
                "timeout": 600,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=400,
                func=embedding_func,
            ),
            embedding_func_max_async=1,
            llm_model_max_async=1,
            default_embedding_timeout=600,
            default_llm_timeout=600,
            chunk_token_size=100,
            chunk_overlap_token_size=10,
        )

        logger.info("LightRAG initialized")
        return rag

    async def insert_document(self, text: str, description: str = "") -> Dict[str, Any]:
        """Insert a document into the knowledge graph."""
        from lightrag.kg.shared_storage import initialize_pipeline_status

        logger.info(f"Inserting document: {description or 'Unnamed'} ({len(text)} chars)")

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

        await self.rag.ainsert(text)

        logger.info("Document inserted successfully")
        return {'status': 'success', 'description': description, 'text_length': len(text)}

    async def query_local(self, query: str) -> str:
        """Local query using graph structure."""
        from lightrag import QueryParam
        logger.info(f"Query: {query}")
        return await self.rag.aquery(query=query, param=QueryParam(mode="local"))

    def export_graph_stats(self) -> Dict[str, Any]:
        """Get KG statistics."""
        graph_files = list(self.working_dir.glob("*.json"))
        kv_files = list(self.working_dir.glob("kv_store_*.json"))
        return {
            'working_dir': str(self.working_dir),
            'total_files': len(graph_files) + len(kv_files),
            'files': [f.name for f in sorted(graph_files + kv_files)],
        }
