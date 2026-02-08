"""
API Bridge for kg-builder pipeline.

Wraps CLI functions (_async_build, generate, train) as importable async functions
with ProgressCallback support for real-time event streaming to the admin UI.

Usage:
    from src.api_bridge import run_build, run_generate, run_train, ProgressCallback

    async def my_callback(event: dict):
        await save_to_database(event)

    result = await run_build(config_dict, callback=my_callback)
"""

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .utils.config import ProjectConfig, FinetuneConfig, TrainingConfig, load_config

logger = logging.getLogger(__name__)

# Type alias for progress callbacks
ProgressCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


def config_from_dict(data: dict) -> ProjectConfig:
    """Create a ProjectConfig from a JSONB dict (as stored in Supabase)."""
    config = ProjectConfig()

    project = data.get("project", {})
    if project.get("name"):
        config.project_name = project["name"]

    chunking = data.get("chunking", {})
    if chunking.get("profile"):
        config.chunking_profile = chunking["profile"]
    if chunking.get("max_chunk_size"):
        config.max_chunk_size = int(chunking["max_chunk_size"])
    if chunking.get("min_chunk_size"):
        config.min_chunk_size = int(chunking["min_chunk_size"])

    llm = data.get("llm", {})
    if llm.get("provider"):
        config.llm_provider = llm["provider"]
    if llm.get("vllm_url"):
        config.vllm_url = llm["vllm_url"]
    if llm.get("ollama_url"):
        config.ollama_url = llm["ollama_url"]
    if llm.get("model"):
        config.llm_model = llm["model"]
    if llm.get("embedding_model"):
        config.embedding_model = llm["embedding_model"]
    if llm.get("embedding_dim"):
        config.embedding_dim = int(llm["embedding_dim"])

    paths = data.get("paths", {})
    if paths.get("sources"):
        config.sources_dir = paths["sources"]
    if paths.get("output"):
        config.output_dir = paths["output"]

    extraction = data.get("extraction", {})
    if extraction.get("cleanup_patterns"):
        config.cleanup_patterns = extraction["cleanup_patterns"]

    ft = data.get("finetune", {})
    if ft:
        if ft.get("system_prompt"):
            config.finetune.system_prompt = ft["system_prompt"]
        if ft.get("strategies"):
            config.finetune.strategies = ft["strategies"]
        if ft.get("format"):
            config.finetune.format = ft["format"]
        filters = ft.get("filters", {})
        if filters.get("min_answer_length") is not None:
            config.finetune.min_answer_length = int(filters["min_answer_length"])
        if filters.get("max_answer_length") is not None:
            config.finetune.max_answer_length = int(filters["max_answer_length"])
        if filters.get("deduplicate") is not None:
            config.finetune.deduplicate = bool(filters["deduplicate"])

    tr = data.get("training", {})
    if tr:
        trc = config.training
        for key in [
            "model", "max_seq_length", "load_in_4bit", "lora_r", "lora_alpha",
            "lora_dropout", "epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "warmup_steps", "save_formats",
        ]:
            if tr.get(key) is not None:
                setattr(trc, key, tr[key])

    return config


async def _emit(callback: Optional[ProgressCallback], event: Dict[str, Any]):
    """Emit a progress event if callback is provided."""
    if callback:
        await callback(event)


async def run_build(
    config_dict: dict,
    sources_dir: str,
    output_dir: str,
    hardware_preset: str = "4090",
    num_pipelines: Optional[int] = None,
    test_chunks: Optional[int] = None,
    skip_merge: bool = False,
    clean: bool = False,
    callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Run the KG build pipeline with progress callbacks.

    Args:
        config_dict: Project config as dict (from Supabase JSONB)
        sources_dir: Path to source PDFs
        output_dir: Path to KG output
        hardware_preset: Hardware preset name (4090, dual4090, etc.)
        num_pipelines: Override pipeline count
        test_chunks: Limit to N chunks for testing
        skip_merge: Skip graph merging
        clean: Force fresh start
        callback: Async callback receiving progress events

    Returns:
        Dict with build results
    """
    from .cli import PRESETS, check_vllm_health, check_ollama_health
    from .cli import load_or_extract_text, load_or_create_chunks
    from .kg.parallel import ParallelKGBuilder, ParallelConfig
    from .kg.merger import merge_pipeline_outputs, get_graph_stats

    config = config_from_dict(config_dict)
    sources_path = Path(sources_dir)
    output_path = Path(output_dir)

    preset = PRESETS.get(hardware_preset, {})
    pipelines = num_pipelines or preset.get("pipelines", 2)

    output_path.mkdir(parents=True, exist_ok=True)

    if clean and output_path.exists():
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # Phase: Health checks
    await _emit(callback, {
        "phase": "health_check", "action": "start", "details": {}
    })

    vllm_ok = await check_vllm_health(config.vllm_url)
    if not vllm_ok:
        await _emit(callback, {
            "phase": "health_check", "action": "error",
            "details": {"error": f"vLLM server not available at {config.vllm_url}"}
        })
        return {"success": False, "error": "vLLM server not available"}

    ollama_ok = await check_ollama_health(config.ollama_url, config.embedding_model)
    if not ollama_ok:
        await _emit(callback, {
            "phase": "health_check", "action": "error",
            "details": {"error": f"Ollama not available at {config.ollama_url}"}
        })
        return {"success": False, "error": "Ollama not available"}

    await _emit(callback, {
        "phase": "health_check", "action": "complete", "details": {}
    })

    # Phase: PDF Extraction
    await _emit(callback, {
        "phase": "pdf_extraction", "action": "start",
        "details": {"sources_dir": str(sources_path)}
    })

    text = load_or_extract_text(
        sources_path, output_path,
        cleanup_patterns=config.cleanup_patterns or None,
        force=clean,
    )

    await _emit(callback, {
        "phase": "pdf_extraction", "action": "complete",
        "details": {"chars": len(text)}
    })

    # Phase: Chunking
    await _emit(callback, {
        "phase": "chunking", "action": "start",
        "details": {"profile": config.chunking_profile}
    })

    chunks = load_or_create_chunks(
        text, config.chunking_profile, output_path,
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
        force=clean,
    )

    total_chunks = len(chunks)

    if test_chunks:
        chunks = chunks[:test_chunks]

    await _emit(callback, {
        "phase": "chunking", "action": "complete",
        "details": {"total_chunks": total_chunks, "used_chunks": len(chunks)}
    })

    # Phase: Parallel KG Build
    await _emit(callback, {
        "phase": "kg_build", "action": "start",
        "details": {"pipelines": pipelines, "chunks": len(chunks)}
    })

    parallel_config = ParallelConfig(
        num_pipelines=pipelines,
        base_output_dir=output_path,
        vllm_url=config.vllm_url,
        ollama_url=config.ollama_url,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        llm_model=config.llm_model,
    )

    kg_start = time.perf_counter()
    builder = ParallelKGBuilder(parallel_config)
    results = await builder.run_parallel(chunks)
    kg_duration_ms = int((time.perf_counter() - kg_start) * 1000)

    pipeline_results = []
    for r in results:
        pipeline_results.append({
            "pipeline_id": r.pipeline_id,
            "chunks_processed": r.chunks_processed,
            "chunks_total": r.chunks_total,
            "duration_ms": r.duration_ms,
            "success": r.success,
            "error": r.error,
        })

    await _emit(callback, {
        "phase": "kg_build", "action": "complete",
        "details": {
            "duration_ms": kg_duration_ms,
            "pipelines": pipeline_results,
        }
    })

    # Phase: Merge
    merged_graph = None
    graph_stats = None
    if not skip_merge and all(r.success for r in results):
        await _emit(callback, {
            "phase": "merge", "action": "start", "details": {}
        })

        merged_dir = output_path / "merged"
        merged_graph = merge_pipeline_outputs(
            [str(d) for d in builder.get_output_dirs()],
            str(merged_dir)
        )
        graph_stats = get_graph_stats(merged_graph)

        await _emit(callback, {
            "phase": "merge", "action": "complete",
            "details": {"graph_path": str(merged_graph), "stats": graph_stats}
        })

    total_time_ms = int((time.perf_counter() - start_time) * 1000)
    success = all(r.success for r in results)

    return {
        "success": success,
        "total_time_ms": total_time_ms,
        "chunks_total": len(chunks),
        "chunks_processed": sum(r.chunks_processed for r in results),
        "pipeline_results": pipeline_results,
        "merged_graph": str(merged_graph) if merged_graph else None,
        "graph_stats": graph_stats,
        "error": None if success else "Some pipelines failed",
    }


async def run_generate(
    config_dict: dict,
    output_dir: str,
    callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Run fine-tuning pair generation with progress callbacks.

    Args:
        config_dict: Project config as dict
        output_dir: KG output directory (where graph + KV stores live)
        callback: Async callback receiving progress events

    Returns:
        Dict with generation results
    """
    from .finetune.generator import FinetuneGenerator
    from .finetune.filters import FilterConfig

    config = config_from_dict(config_dict)
    ftc = config.finetune
    output_path = Path(output_dir)

    await _emit(callback, {
        "phase": "finetune_generate", "action": "start",
        "details": {"strategies": ftc.strategies, "format": ftc.format}
    })

    filter_config = FilterConfig(
        min_answer_length=ftc.min_answer_length,
        max_answer_length=ftc.max_answer_length,
        min_question_length=ftc.min_question_length,
        deduplicate=ftc.deduplicate,
    )

    generator = FinetuneGenerator(
        kg_dir=output_path,
        output_dir=output_path,
        strategy_names=ftc.strategies,
        format_name=ftc.format,
        system_prompt=ftc.system_prompt,
        filter_config=filter_config,
        augment_n=0,
        augment_llm=False,
        llm_url=config.vllm_url,
        llm_model=config.llm_model,
    )

    report = generator.run()

    result = {
        "success": True,
        "total_generated": report.total_generated,
        "total_accepted": report.total_accepted,
        "total_rejected": report.total_rejected,
        "per_strategy": report.per_strategy,
    }

    await _emit(callback, {
        "phase": "finetune_generate", "action": "complete",
        "details": result,
    })

    return result


async def run_train(
    config_dict: dict,
    dataset_path: str,
    output_dir: str,
    callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Run LoRA fine-tuning with progress callbacks.

    Args:
        config_dict: Project config as dict
        dataset_path: Path to OpenAI JSONL file
        output_dir: Output directory for trained model
        callback: Async callback receiving progress events

    Returns:
        Dict with training results
    """
    from .finetune.trainer import LoRATrainer

    config = config_from_dict(config_dict)
    trc = config.training

    await _emit(callback, {
        "phase": "training", "action": "start",
        "details": {
            "model": trc.model,
            "epochs": trc.epochs,
            "batch_size": trc.batch_size,
            "lora_r": trc.lora_r,
        }
    })

    trainer = LoRATrainer(config=trc, output_dir=Path(output_dir))
    report = trainer.run(dataset_path=Path(dataset_path))

    result = {
        "success": True,
        "duration_min": report.get("duration_min"),
        "final_loss": report.get("final_loss"),
    }

    await _emit(callback, {
        "phase": "training", "action": "complete",
        "details": result,
    })

    return result
