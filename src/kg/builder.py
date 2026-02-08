"""
Knowledge Graph Builder

Core LightRAG builder that integrates with Ollama for entity extraction
and graph building. Configurable for any domain (no hardcoded defaults).
"""

import logging
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
        self._embedding_func = None

        self.rag = self._init_lightrag()

    def _init_lightrag(self):
        """Initialize LightRAG with Ollama backend."""
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_model_complete
        from lightrag.utils import EmbeddingFunc
        from ..llm.embedding import create_ollama_embedding_func

        logger.info(f"Configuring LightRAG - LLM: {self.llm_model}, Embedding: {self.embedding_model}")

        self._embedding_func = create_ollama_embedding_func(
            ollama_url=self.ollama_url,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
            timeout=300.0,
            max_retries=5,
            retry_base_delay=2.0,
            inter_request_delay=0.5,
        )

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
                func=self._embedding_func,
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

    async def close(self):
        """Close the embedding HTTP client."""
        if self._embedding_func is not None and hasattr(self._embedding_func, 'close'):
            await self._embedding_func.close()

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
