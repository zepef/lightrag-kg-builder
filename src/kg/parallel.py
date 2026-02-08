"""
Parallel KG Pipeline Orchestrator

Orchestrates N parallel LightRAG pipelines, each processing a subset of chunks.
Designed for multi-GPU setups or high-throughput vLLM configurations.

Architecture:
    +-------------------------------------------+
    |  Shared vLLM Server (continuous batch)    |
    +-------------------------------------------+
           ^           ^           ^
    +--------+ +--------+ +--------+
    | Pipe 1 | | Pipe 2 | | Pipe N |
    +--------+ +--------+ +--------+
           |           |           |
           +-----+-----+---------+
                 v
    +-------------------------------------------+
    |  Graph Merger (entity deduplication)      |
    +-------------------------------------------+
"""

import asyncio
import json
import hashlib
import time
import logging
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline worker."""
    pipeline_id: int
    working_dir: Path
    chunk_range: tuple[int, int]
    vllm_url: str
    ollama_url: str
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_tokens: int = 2048
    temperature: float = 0.1


@dataclass
class PipelineResult:
    """Result from a single pipeline worker."""
    pipeline_id: int
    output_dir: Path
    chunks_processed: int
    chunks_total: int
    duration_ms: int
    success: bool
    error: Optional[str] = None


@dataclass
class ParallelConfig:
    """Configuration for parallel pipeline orchestration."""
    num_pipelines: int
    base_output_dir: Path
    vllm_url: str = "http://localhost:8000/v1"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_concurrent_requests_per_pipeline: int = 2
    embedding_delay: float = 0.1
    progress_file: str = "parallel_progress.json"


class PipelineWorker:
    """
    Single pipeline worker that processes a subset of chunks.

    Each worker operates independently with its own LightRAG instance
    and output directory, allowing true parallel processing.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.rag = None
        self.chunks_processed = 0
        self.progress_file = config.working_dir / "progress.json"

    async def initialize(self):
        """Initialize LightRAG with vLLM as LLM backend."""
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        from ..llm.vllm_client import create_vllm_llm_func
        from ..llm.embedding import create_ollama_embedding_func

        self.config.working_dir.mkdir(parents=True, exist_ok=True)

        vllm_complete = create_vllm_llm_func(
            vllm_url=self.config.vllm_url,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        embedding_func = create_ollama_embedding_func(
            ollama_url=self.config.ollama_url,
            embedding_model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
        )

        self.rag = LightRAG(
            working_dir=str(self.config.working_dir),
            llm_model_func=vllm_complete,
            llm_model_name=self.config.llm_model,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.embedding_dim,
                max_token_size=400,
                func=embedding_func,
            ),
            embedding_func_max_async=16,
            llm_model_max_async=32,
            chunk_token_size=100,
            chunk_overlap_token_size=10,
        )

        await self.rag.initialize_storages()
        logger.info(f"Pipeline {self.config.pipeline_id} initialized at {self.config.working_dir}")

    async def process_chunks(self, chunks: List[Dict[str, Any]]) -> PipelineResult:
        """Process assigned chunks through LightRAG."""
        from lightrag.kg.shared_storage import initialize_pipeline_status

        start_time = time.perf_counter()
        await initialize_pipeline_status()

        progress = self._load_progress()
        completed_ids = set(progress.get('completed_chunks', []))

        start_idx, end_idx = self.config.chunk_range
        our_chunks = chunks[start_idx:end_idx]

        pending = [c for c in our_chunks if c['hash_id'] not in completed_ids]

        logger.info(
            f"Pipeline {self.config.pipeline_id}: "
            f"Processing {len(pending)}/{len(our_chunks)} chunks "
            f"(range {start_idx}-{end_idx})"
        )

        PERSIST_EVERY = 5

        try:
            for i, chunk in enumerate(pending):
                chunk_start = time.perf_counter()

                await self.rag.ainsert(chunk['text'])

                completed_ids.add(chunk['hash_id'])
                self.chunks_processed += 1
                chunk_ms = int((time.perf_counter() - chunk_start) * 1000)

                if (i + 1) % PERSIST_EVERY == 0 or (i + 1) == len(pending):
                    await self.rag.finalize_storages()
                    await self.rag.initialize_storages()
                    progress['completed_chunks'] = list(completed_ids)
                    progress['last_processed'] = chunk['hash_id']
                    self._save_progress(progress)

                if (i + 1) % 5 == 0:
                    logger.info(
                        f"Pipeline {self.config.pipeline_id}: "
                        f"{len(completed_ids)}/{len(our_chunks)} chunks, "
                        f"last chunk: {chunk_ms}ms"
                    )

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return PipelineResult(
                pipeline_id=self.config.pipeline_id,
                output_dir=self.config.working_dir,
                chunks_processed=len(completed_ids),
                chunks_total=len(our_chunks),
                duration_ms=duration_ms,
                success=True
            )

        except Exception as e:
            logger.error(f"Pipeline {self.config.pipeline_id} failed: {e}")
            try:
                await self.rag.finalize_storages()
            except Exception:
                pass

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return PipelineResult(
                pipeline_id=self.config.pipeline_id,
                output_dir=self.config.working_dir,
                chunks_processed=len(completed_ids),
                chunks_total=len(our_chunks),
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            return json.loads(self.progress_file.read_text())
        return {'completed_chunks': []}

    def _save_progress(self, progress: Dict[str, Any]):
        self.progress_file.write_text(json.dumps(progress, indent=2))


def _run_pipeline_process(config_dict: Dict, chunks: List[Dict], result_file: str):
    """
    Standalone function that runs a single pipeline in its own process.

    Each process gets its own asyncio event loop, so LightRAG's internal
    async operations don't block other pipelines.
    """
    async def _run():
        config = PipelineConfig(
            pipeline_id=config_dict['pipeline_id'],
            working_dir=Path(config_dict['working_dir']),
            chunk_range=tuple(config_dict['chunk_range']),
            vllm_url=config_dict['vllm_url'],
            ollama_url=config_dict['ollama_url'],
            embedding_model=config_dict.get('embedding_model', 'nomic-embed-text'),
            embedding_dim=config_dict.get('embedding_dim', 768),
            llm_model=config_dict.get('llm_model', 'mistralai/Mistral-7B-Instruct-v0.3'),
        )
        worker = PipelineWorker(config)
        await worker.initialize()
        result = await worker.process_chunks(chunks)
        result_data = {
            'pipeline_id': result.pipeline_id,
            'output_dir': str(result.output_dir),
            'chunks_processed': result.chunks_processed,
            'chunks_total': result.chunks_total,
            'duration_ms': result.duration_ms,
            'success': result.success,
            'error': result.error,
        }
        Path(result_file).write_text(json.dumps(result_data))

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [P{config_dict["pipeline_id"]}] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'/tmp/lightrag_p{config_dict["pipeline_id"]}.log', mode='a')
        ]
    )
    asyncio.run(_run())


class ParallelKGBuilder:
    """
    Orchestrates N parallel LightRAG pipelines.

    Splits chunks into N ranges, runs pipelines concurrently as separate
    OS processes, and collects results for merging.
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.workers: List[PipelineWorker] = []

    def split_chunks(self, chunks: List[Dict[str, Any]]) -> List[tuple[int, int]]:
        """Split chunks into N balanced ranges."""
        n = len(chunks)
        size = n // self.config.num_pipelines
        remainder = n % self.config.num_pipelines

        ranges = []
        start = 0
        for i in range(self.config.num_pipelines):
            end = start + size + (1 if i < remainder else 0)
            ranges.append((start, end))
            start = end

        return ranges

    async def run_parallel(self, chunks: List[Dict[str, Any]]) -> List[PipelineResult]:
        """
        Run N parallel pipelines as separate OS processes.

        Args:
            chunks: Full list of chunks with 'hash_id' and 'text'

        Returns:
            List of PipelineResult from each worker
        """
        for chunk in chunks:
            if 'hash_id' not in chunk:
                chunk['hash_id'] = hashlib.md5(chunk['text'][:500].encode()).hexdigest()[:12]

        ranges = self.split_chunks(chunks)

        logger.info(f"Starting {self.config.num_pipelines} parallel pipelines (multiprocessing)")
        for i, (start, end) in enumerate(ranges):
            logger.info(f"  Pipeline {i+1}: chunks {start}-{end} ({end-start} chunks)")

        processes = []
        result_files = []
        start_time = time.perf_counter()

        for i, (start, end) in enumerate(ranges):
            config_dict = {
                'pipeline_id': i + 1,
                'working_dir': str(self.config.base_output_dir / f"p{i+1}"),
                'chunk_range': [start, end],
                'vllm_url': self.config.vllm_url,
                'ollama_url': self.config.ollama_url,
                'embedding_model': self.config.embedding_model,
                'embedding_dim': self.config.embedding_dim,
                'llm_model': self.config.llm_model,
            }
            result_file = f"/tmp/pipeline_result_p{i+1}.json"
            result_files.append(result_file)

            p = multiprocessing.Process(
                target=_run_pipeline_process,
                args=(config_dict, chunks, result_file),
                name=f"pipeline-{i+1}",
            )
            processes.append(p)

        logger.info(f"Launching {len(processes)} worker processes...")
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        total_ms = int((time.perf_counter() - start_time) * 1000)

        pipeline_results = []
        for i, result_file in enumerate(result_files):
            try:
                data = json.loads(Path(result_file).read_text())
                pipeline_results.append(PipelineResult(
                    pipeline_id=data['pipeline_id'],
                    output_dir=Path(data['output_dir']),
                    chunks_processed=data['chunks_processed'],
                    chunks_total=data['chunks_total'],
                    duration_ms=data['duration_ms'],
                    success=data['success'],
                    error=data.get('error'),
                ))
            except Exception as e:
                logger.error(f"Failed to read result for pipeline {i+1}: {e}")
                pipeline_results.append(PipelineResult(
                    pipeline_id=i + 1,
                    output_dir=self.config.base_output_dir / f"p{i+1}",
                    chunks_processed=0,
                    chunks_total=0,
                    duration_ms=0,
                    success=False,
                    error=f"Process failed: {e}",
                ))

        total_processed = sum(r.chunks_processed for r in pipeline_results)
        total_chunks = sum(r.chunks_total for r in pipeline_results)
        failed = [r for r in pipeline_results if not r.success]

        logger.info(f"\nParallel processing complete:")
        logger.info(f"  Total time: {total_ms}ms ({total_ms/1000:.1f}s)")
        logger.info(f"  Chunks processed: {total_processed}/{total_chunks}")
        logger.info(f"  Failed pipelines: {len(failed)}")

        for r in pipeline_results:
            status = "OK" if r.success else "FAILED"
            logger.info(
                f"  [{status}] Pipeline {r.pipeline_id}: "
                f"{r.chunks_processed}/{r.chunks_total} in {r.duration_ms}ms"
            )
            if r.error:
                logger.info(f"      Error: {r.error}")

        return pipeline_results

    def get_output_dirs(self) -> List[Path]:
        """Get list of pipeline output directories."""
        return [
            self.config.base_output_dir / f"p{i+1}"
            for i in range(self.config.num_pipelines)
        ]
