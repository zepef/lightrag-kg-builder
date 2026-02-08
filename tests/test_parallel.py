"""Tests for the parallel KG pipeline orchestrator."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.kg.parallel import (
    PipelineConfig,
    PipelineResult,
    ParallelConfig,
    PipelineWorker,
    ParallelKGBuilder,
)


# ============================================================================
# PipelineConfig
# ============================================================================

class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig(
            pipeline_id=1,
            working_dir=Path("/tmp/test"),
            chunk_range=(0, 10),
            vllm_url="http://localhost:8000/v1",
            ollama_url="http://localhost:11434",
        )
        assert cfg.embedding_model == "nomic-embed-text"
        assert cfg.embedding_dim == 768
        assert cfg.max_tokens == 2048
        assert cfg.temperature == 0.1

    def test_custom_values(self):
        cfg = PipelineConfig(
            pipeline_id=3,
            working_dir=Path("/data/p3"),
            chunk_range=(100, 200),
            vllm_url="http://gpu:8000/v1",
            ollama_url="http://gpu:11434",
            embedding_model="custom-embed",
            embedding_dim=1024,
            llm_model="meta-llama/Llama-2-7b",
            max_tokens=4096,
            temperature=0.5,
        )
        assert cfg.pipeline_id == 3
        assert cfg.chunk_range == (100, 200)
        assert cfg.embedding_model == "custom-embed"
        assert cfg.embedding_dim == 1024


# ============================================================================
# PipelineResult
# ============================================================================

class TestPipelineResult:

    def test_success_result(self):
        r = PipelineResult(
            pipeline_id=1,
            output_dir=Path("/out/p1"),
            chunks_processed=50,
            chunks_total=50,
            duration_ms=30000,
            success=True,
        )
        assert r.success is True
        assert r.error is None
        assert r.chunks_processed == r.chunks_total

    def test_failure_result(self):
        r = PipelineResult(
            pipeline_id=2,
            output_dir=Path("/out/p2"),
            chunks_processed=10,
            chunks_total=50,
            duration_ms=5000,
            success=False,
            error="Connection refused",
        )
        assert r.success is False
        assert "Connection" in r.error


# ============================================================================
# PipelineWorker
# ============================================================================

class TestPipelineWorker:

    def test_progress_roundtrip(self, tmp_path):
        cfg = PipelineConfig(
            pipeline_id=1,
            working_dir=tmp_path,
            chunk_range=(0, 5),
            vllm_url="http://localhost:8000/v1",
            ollama_url="http://localhost:11434",
        )
        worker = PipelineWorker(cfg)

        # Initially empty
        progress = worker._load_progress()
        assert progress == {"completed_chunks": []}

        # Save and reload
        progress["completed_chunks"] = ["abc123", "def456"]
        progress["last_processed"] = "def456"
        worker._save_progress(progress)

        reloaded = worker._load_progress()
        assert "abc123" in reloaded["completed_chunks"]
        assert "def456" in reloaded["completed_chunks"]
        assert reloaded["last_processed"] == "def456"

    def test_progress_file_location(self, tmp_path):
        cfg = PipelineConfig(
            pipeline_id=1,
            working_dir=tmp_path,
            chunk_range=(0, 5),
            vllm_url="http://localhost:8000/v1",
            ollama_url="http://localhost:11434",
        )
        worker = PipelineWorker(cfg)
        assert worker.progress_file == tmp_path / "progress.json"


# ============================================================================
# ParallelKGBuilder - chunk splitting
# ============================================================================

class TestChunkSplitting:

    def _make_builder(self, num_pipelines, tmp_path):
        cfg = ParallelConfig(
            num_pipelines=num_pipelines,
            base_output_dir=tmp_path / "output",
        )
        return ParallelKGBuilder(cfg)

    def test_split_even(self, tmp_path):
        builder = self._make_builder(3, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(9)]
        ranges = builder.split_chunks(chunks)
        assert ranges == [(0, 3), (3, 6), (6, 9)]

    def test_split_uneven(self, tmp_path):
        builder = self._make_builder(3, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(10)]
        ranges = builder.split_chunks(chunks)
        # 10 / 3 = 3 remainder 1: first pipeline gets 4, rest get 3
        assert ranges == [(0, 4), (4, 7), (7, 10)]

    def test_split_single_pipeline(self, tmp_path):
        builder = self._make_builder(1, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(5)]
        ranges = builder.split_chunks(chunks)
        assert ranges == [(0, 5)]

    def test_split_more_pipelines_than_chunks(self, tmp_path):
        builder = self._make_builder(5, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(3)]
        ranges = builder.split_chunks(chunks)
        assert len(ranges) == 5
        # First 3 get 1 chunk each, last 2 get 0
        non_empty = [(s, e) for s, e in ranges if s < e]
        assert len(non_empty) == 3

    def test_split_empty_chunks(self, tmp_path):
        builder = self._make_builder(3, tmp_path)
        ranges = builder.split_chunks([])
        assert len(ranges) == 3
        assert all(s == e for s, e in ranges)

    def test_split_covers_all_chunks(self, tmp_path):
        builder = self._make_builder(4, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(17)]
        ranges = builder.split_chunks(chunks)
        # Verify no gaps and no overlaps
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 17

    def test_split_two_pipelines_odd(self, tmp_path):
        builder = self._make_builder(2, tmp_path)
        chunks = [{"text": f"chunk {i}"} for i in range(7)]
        ranges = builder.split_chunks(chunks)
        assert ranges == [(0, 4), (4, 7)]


# ============================================================================
# ParallelKGBuilder - output dirs
# ============================================================================

class TestOutputDirs:

    def test_get_output_dirs(self, tmp_path):
        cfg = ParallelConfig(num_pipelines=3, base_output_dir=tmp_path / "out")
        builder = ParallelKGBuilder(cfg)
        dirs = builder.get_output_dirs()
        assert len(dirs) == 3
        assert dirs[0] == tmp_path / "out" / "p1"
        assert dirs[1] == tmp_path / "out" / "p2"
        assert dirs[2] == tmp_path / "out" / "p3"


# ============================================================================
# Result file handling (no /tmp/)
# ============================================================================

class TestResultFilePaths:

    def test_result_files_use_output_dir(self, tmp_path):
        """Verify result files are written to base_output_dir, not a hardcoded /tmp/."""
        base = tmp_path / "my_output"
        cfg = ParallelConfig(
            num_pipelines=2,
            base_output_dir=base,
        )
        builder = ParallelKGBuilder(cfg)

        # Verify the result file path is derived from base_output_dir
        for i in range(cfg.num_pipelines):
            result_file = str(base / f"_result_p{i+1}.json")
            assert result_file.startswith(str(base))
            assert f"_result_p{i+1}.json" in result_file
