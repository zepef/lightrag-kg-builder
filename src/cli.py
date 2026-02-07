#!/usr/bin/env python3
"""
lightrag-kg CLI

Command-line interface for building Knowledge Graphs with LightRAG.

Usage:
    python -m src.cli build --config configs/pcg2026.yaml --sources /path/to/pdfs --output /path/to/kg
    python -m src.cli build --config configs/pcg2026.yaml --mode 8x5090
    python -m src.cli status --output /path/to/kg
    python -m src.cli health --vllm-url http://localhost:8000/v1
"""

import asyncio
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import click

from .utils.config import load_config, merge_cli_overrides, ProjectConfig, FinetuneConfig
from .utils.journal import ExecutionJournal
from .extractors.pdf_extractor import PdfExtractor
from .kg.parallel import ParallelKGBuilder, ParallelConfig
from .kg.merger import merge_pipeline_outputs, get_graph_stats

logger = logging.getLogger(__name__)

# ============================================================================
# Hardware Presets
# ============================================================================

PRESETS = {
    "4090": {
        "name": "RTX 4090 (24GB)",
        "pipelines": 2,
        "description": "2 parallel pipelines sharing 1 vLLM server (~18GB VRAM)",
        "estimated_speedup": "~2x",
    },
    "dual4090": {
        "name": "2x RTX 4090 + 128GB RAM",
        "pipelines": 4,
        "description": "4 parallel pipelines, 2-way tensor parallel vLLM (48GB VRAM)",
        "estimated_speedup": "~4x",
    },
    "8x5090": {
        "name": "8x RTX 5090 (256GB VRAM)",
        "pipelines": 16,
        "description": "16 parallel pipelines, 8-way tensor parallel vLLM",
        "estimated_speedup": "~12x",
    },
    "bigpu": {
        "name": "8x A100/H100",
        "pipelines": 4,
        "description": "4 parallel pipelines with tensor-parallel vLLM",
        "estimated_speedup": "~4x",
    },
    "max": {
        "name": "8x H100 NVLink Cluster",
        "pipelines": 16,
        "description": "16 parallel pipelines, 8-way tensor parallel vLLM",
        "estimated_speedup": "~10x",
    },
}


# ============================================================================
# Chunking Profile Registry
# ============================================================================

def get_chunker(profile_name: str, max_chunk_size: int = 4000, min_chunk_size: int = 500):
    """Get a chunker for the given profile name."""
    from .chunkers.legal_chunker import LegalDocumentChunker

    if profile_name == "pcg":
        from .chunkers.profiles.pcg import PCG_PROFILE
        return LegalDocumentChunker(PCG_PROFILE, max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size)
    else:
        raise ValueError(f"Unknown chunking profile: {profile_name}. Available: pcg")


# ============================================================================
# Health Checks
# ============================================================================

async def check_vllm_health(url: str) -> bool:
    """Check if vLLM server is healthy."""
    from .llm.vllm_client import test_vllm_connection
    logger.info(f"Checking vLLM server at {url}...")
    try:
        return await test_vllm_connection(url)
    except Exception as e:
        logger.error(f"vLLM health check failed: {e}")
        return False


async def check_ollama_health(url: str, model: str = "nomic-embed-text") -> bool:
    """Check if Ollama is healthy and model is available."""
    import httpx
    logger.info(f"Checking Ollama at {url} for model {model}...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code != 200:
                logger.error("Ollama not responding")
                return False

            models = resp.json().get("models", [])
            model_names = [m["name"].split(":")[0] for m in models]
            if model not in model_names and f"{model}:latest" not in [m["name"] for m in models]:
                logger.warning(f"Model {model} not found. Available: {model_names}")
                logger.info(f"Attempting to pull {model}...")
                pull_resp = await client.post(
                    f"{url}/api/pull",
                    json={"name": model},
                    timeout=300.0
                )
                if pull_resp.status_code != 200:
                    logger.error(f"Failed to pull {model}")
                    return False

            logger.info("Ollama healthy and model available")
            return True
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return False


# ============================================================================
# Data Preparation
# ============================================================================

def load_or_extract_text(
    sources_dir: Path,
    output_dir: Path,
    cleanup_patterns: Optional[list] = None,
    force: bool = False,
) -> str:
    """Load cached extracted text or extract from PDFs."""
    cached_path = output_dir / "extracted_text.txt"

    if cached_path.exists() and not force:
        logger.info(f"Using cached extraction: {cached_path}")
        return cached_path.read_text(encoding='utf-8')

    logger.info(f"Extracting text from PDFs in {sources_dir}...")
    extractor = PdfExtractor(cleanup_patterns=cleanup_patterns or None)
    result = extractor.extract_all(sources_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(result.text, encoding='utf-8')
    logger.info(f"Extracted {result.chars:,} chars from {result.pages} pages")

    return result.text


def load_or_create_chunks(
    text: str,
    profile_name: str,
    output_dir: Path,
    max_chunk_size: int = 4000,
    min_chunk_size: int = 500,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Load cached chunks or create from text."""
    chunks_cache = output_dir / "chunks_cache.json"

    if chunks_cache.exists() and not force:
        try:
            chunks = json.loads(chunks_cache.read_text())
            logger.info(f"Using cached chunks: {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.warning(f"Failed to load chunks cache: {e}")

    logger.info(f"Creating semantic chunks with profile '{profile_name}'...")
    chunker = get_chunker(profile_name, max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size)
    chunks = chunker.chunk_document(text)

    # Add hash IDs
    for chunk in chunks:
        chunk['hash_id'] = hashlib.md5(chunk['text'][:500].encode()).hexdigest()[:12]

    # Cache
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_cache.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    logger.info(f"Created and cached {len(chunks)} chunks")

    return chunks


# ============================================================================
# Display Helpers
# ============================================================================

def print_banner(project_name: str, mode: str, num_pipelines: int):
    """Print startup banner."""
    preset = PRESETS.get(mode, {})
    print("\n" + "=" * 70)
    print(f"  {project_name} - Knowledge Graph Builder")
    print("=" * 70)
    print(f"  Mode:       {preset.get('name', mode)}")
    print(f"  Pipelines:  {num_pipelines}")
    print(f"  Speedup:    {preset.get('estimated_speedup', 'N/A')}")
    print(f"  Config:     {preset.get('description', 'Custom')}")
    print("=" * 70 + "\n")


def print_results(results, total_time_ms: int, merged_graph: Optional[Path]):
    """Print final results summary."""
    print("\n" + "=" * 70)
    print("  BUILD RESULTS")
    print("=" * 70)

    total_chunks = sum(r.chunks_total for r in results)
    total_processed = sum(r.chunks_processed for r in results)
    failed = len([r for r in results if not r.success])

    print(f"\n  Total Time:       {total_time_ms/1000:.1f}s ({total_time_ms/60000:.1f} min)")
    print(f"  Chunks Processed: {total_processed}/{total_chunks}")
    print(f"  Failed Pipelines: {failed}")

    print("\n  Pipeline Breakdown:")
    for r in results:
        status = "OK" if r.success else "FAILED"
        rate = r.chunks_processed / (r.duration_ms / 60000) if r.duration_ms > 0 else 0
        print(
            f"    [{status}] Pipeline {r.pipeline_id}: "
            f"{r.chunks_processed}/{r.chunks_total} chunks, "
            f"{r.duration_ms/1000:.1f}s ({rate:.1f} chunks/min)"
        )

    if merged_graph:
        stats = get_graph_stats(merged_graph)
        print(f"\n  Merged Graph: {merged_graph}")
        print(f"    Nodes: {stats.get('nodes', 'N/A')}")
        print(f"    Edges: {stats.get('edges', 'N/A')}")

    print("\n" + "=" * 70)


# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """lightrag-kg-builder: Build Knowledge Graphs from documents using LightRAG."""
    pass


@cli.command()
@click.option('--config', 'config_path', required=True, help='Path to YAML config file')
@click.option('--sources', help='Source documents directory (overrides config)')
@click.option('--output', help='KG output directory (overrides config)')
@click.option('--mode', type=click.Choice(['4090', 'dual4090', '8x5090', 'bigpu', 'max']),
              default='4090', help='Hardware configuration preset')
@click.option('--pipelines', type=int, default=None, help='Override number of parallel pipelines')
@click.option('--vllm-url', default=None, help='vLLM server URL (overrides config)')
@click.option('--ollama-url', default=None, help='Ollama server URL (overrides config)')
@click.option('--test-chunks', type=int, default=None, help='Only process N chunks (for testing)')
@click.option('--resume/--no-resume', default=False, help='Resume from last checkpoint')
@click.option('--clean', is_flag=True, help='Clear all data and start fresh')
@click.option('--skip-merge', is_flag=True, help='Skip graph merging step')
@click.option('--skip-health-check', is_flag=True, help='Skip vLLM/Ollama health checks')
def build(config_path, sources, output, mode, pipelines, vllm_url, ollama_url,
          test_chunks, resume, clean, skip_merge, skip_health_check):
    """Build a Knowledge Graph from source documents."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/tmp/lightrag_kg_build.log', mode='a')
        ]
    )

    # Load config
    config = load_config(config_path)
    config = merge_cli_overrides(
        config,
        vllm_url=vllm_url,
        ollama_url=ollama_url,
        sources_dir=sources,
        output_dir=output,
    )

    # Resolve paths
    sources_dir = Path(config.sources_dir) if config.sources_dir else None
    output_dir = Path(config.output_dir) if config.output_dir else Path("./output/kg")

    if not sources_dir:
        click.echo("Error: --sources is required (or set paths.sources in config)")
        sys.exit(1)

    # Determine pipeline count
    preset = PRESETS.get(mode, {})
    num_pipelines = pipelines or preset.get('pipelines', 2)

    print_banner(config.project_name, mode, num_pipelines)

    print(f"  Sources:    {sources_dir}")
    print(f"  Output:     {output_dir}")

    # Handle --clean
    if clean:
        print("\n  Cleaning output data...")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("  Clean complete")

    # Run async build
    exit_code = asyncio.run(_async_build(
        config=config,
        sources_dir=sources_dir,
        output_dir=output_dir,
        num_pipelines=num_pipelines,
        mode=mode,
        test_chunks=test_chunks,
        skip_merge=skip_merge,
        skip_health_check=skip_health_check,
        clean=clean,
    ))
    sys.exit(exit_code)


async def _async_build(
    config: ProjectConfig,
    sources_dir: Path,
    output_dir: Path,
    num_pipelines: int,
    mode: str,
    test_chunks: Optional[int],
    skip_merge: bool,
    skip_health_check: bool,
    clean: bool,
) -> int:
    """Async build orchestration."""
    # Health checks
    if not skip_health_check:
        print("\n  Running health checks...")

        vllm_ok = await check_vllm_health(config.vllm_url)
        if not vllm_ok:
            print("\n  vLLM server not available!")
            print("\n  Start vLLM server first. See README.md for instructions.")
            return 1

        ollama_ok = await check_ollama_health(config.ollama_url, config.embedding_model)
        if not ollama_ok:
            print("\n  Ollama not available!")
            print(f"\n  Please ensure Ollama is running with {config.embedding_model} model:")
            print(f"    ollama pull {config.embedding_model}")
            print("    ollama serve")
            return 1

        print("  All services healthy\n")

    # Load/extract text
    print("  Phase 1: Loading source text...")
    text = load_or_extract_text(
        sources_dir, output_dir,
        cleanup_patterns=config.cleanup_patterns or None,
        force=clean,
    )
    print(f"    {len(text):,} characters\n")

    # Create chunks
    print("  Phase 2: Creating semantic chunks...")
    chunks = load_or_create_chunks(
        text, config.chunking_profile, output_dir,
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
        force=clean,
    )
    print(f"    {len(chunks)} chunks\n")

    # Limit chunks for testing
    if test_chunks:
        chunks = chunks[:test_chunks]
        print(f"    Test mode: Using only {len(chunks)} chunks\n")

    # Setup parallel config
    parallel_config = ParallelConfig(
        num_pipelines=num_pipelines,
        base_output_dir=output_dir,
        vllm_url=config.vllm_url,
        ollama_url=config.ollama_url,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        llm_model=config.llm_model,
    )

    # Run parallel build
    print("  Phase 3: Running parallel KG build...")
    start_time = time.perf_counter()

    builder = ParallelKGBuilder(parallel_config)
    results = await builder.run_parallel(chunks)

    total_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Merge if all succeeded
    merged_graph = None
    if not skip_merge and all(r.success for r in results):
        print("\n  Phase 4: Merging graphs...")
        merged_dir = output_dir / "merged"
        merged_graph = merge_pipeline_outputs(
            [str(d) for d in builder.get_output_dirs()],
            str(merged_dir)
        )
        print(f"    Merged graph: {merged_graph}")

    # Print results
    print_results(results, total_time_ms, merged_graph)

    # Save timing report
    report = {
        'project': config.project_name,
        'mode': mode,
        'pipelines': num_pipelines,
        'chunks_total': len(chunks),
        'chunks_processed': sum(r.chunks_processed for r in results),
        'total_time_ms': total_time_ms,
        'total_time_min': round(total_time_ms / 60000, 2),
        'pipeline_results': [
            {
                'id': r.pipeline_id,
                'processed': r.chunks_processed,
                'total': r.chunks_total,
                'duration_ms': r.duration_ms,
                'success': r.success,
                'error': r.error
            }
            for r in results
        ],
        'merged_graph': str(merged_graph) if merged_graph else None,
    }

    report_path = output_dir / "timing_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Timing report saved: {report_path}")

    if all(r.success for r in results):
        print("\n  BUILD COMPLETE")
        return 0
    else:
        print("\n  BUILD FAILED (some pipelines failed)")
        return 1


@cli.command()
@click.option('--output', required=True, help='KG output directory to check')
def status(output):
    """Check build status and graph statistics."""
    output_dir = Path(output)

    if not output_dir.exists():
        click.echo(f"Output directory not found: {output_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  KG Build Status")
    print("=" * 60)

    # Check journal
    journal_path = output_dir / "execution_journal.jsonl"
    if journal_path.exists():
        journal = ExecutionJournal(journal_path)
        summary = journal.summary()
        print(f"\n  Status: {summary['status']}")
        print(f"  Entries: {summary['entries']}")
        print(f"  Errors: {summary['error_count']}")
        if summary.get('phases'):
            print("  Phases:")
            for phase, info in summary['phases'].items():
                status_icon = "OK" if info['completed'] else "IN-PROGRESS" if info['started'] else "PENDING"
                duration = f" ({info['duration_ms']}ms)" if info['duration_ms'] else ""
                print(f"    [{status_icon}] {phase}{duration}")
    else:
        print("  No journal found")

    # Check timing report
    report_path = output_dir / "timing_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"\n  Last Build:")
        print(f"    Project: {report.get('project', 'N/A')}")
        print(f"    Time: {report.get('total_time_min', 'N/A')} min")
        print(f"    Chunks: {report.get('chunks_processed', 0)}/{report.get('chunks_total', 0)}")

    # Check merged graph
    merged_graph = output_dir / "merged" / "graph_chunk_entity_relation.graphml"
    if merged_graph.exists():
        stats = get_graph_stats(merged_graph)
        print(f"\n  Merged Graph: {merged_graph}")
        print(f"    Nodes: {stats.get('nodes', 'N/A')}")
        print(f"    Edges: {stats.get('edges', 'N/A')}")
        print(f"    Density: {stats.get('density', 'N/A')}")
        print(f"    Components: {stats.get('connected_components', 'N/A')}")
    else:
        # Check pipeline dirs
        pipeline_dirs = sorted(output_dir.glob("p*/graph_chunk_entity_relation.graphml"))
        if pipeline_dirs:
            print(f"\n  Pipeline Graphs ({len(pipeline_dirs)} found):")
            for gf in pipeline_dirs:
                stats = get_graph_stats(gf)
                print(f"    {gf.parent.name}: {stats.get('nodes', '?')} nodes, {stats.get('edges', '?')} edges")
        else:
            print("\n  No graph files found")

    print("\n" + "=" * 60)


@cli.command()
@click.option('--vllm-url', default='http://localhost:8000/v1', help='vLLM server URL')
@click.option('--ollama-url', default='http://localhost:11434', help='Ollama server URL')
@click.option('--embedding-model', default='nomic-embed-text', help='Embedding model name')
def health(vllm_url, ollama_url, embedding_model):
    """Check vLLM and Ollama server health."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

    async def _check():
        print("\n  Health Check")
        print("  " + "-" * 40)

        vllm_ok = await check_vllm_health(vllm_url)
        print(f"  vLLM ({vllm_url}): {'OK' if vllm_ok else 'FAILED'}")

        ollama_ok = await check_ollama_health(ollama_url, embedding_model)
        print(f"  Ollama ({ollama_url}): {'OK' if ollama_ok else 'FAILED'}")
        print(f"  Embedding model ({embedding_model}): {'OK' if ollama_ok else 'FAILED'}")

        print("  " + "-" * 40)
        if vllm_ok and ollama_ok:
            print("  All services healthy")
            return 0
        else:
            print("  Some services unhealthy")
            return 1

    exit_code = asyncio.run(_check())
    sys.exit(exit_code)


@cli.command()
@click.option('--config', 'config_path', default=None, help='Path to YAML config file')
@click.option('--output', required=True, help='KG directory (where graph + KV stores live)')
@click.option('--format', 'fmt', type=click.Choice(['openai', 'alpaca', 'sharegpt']),
              default=None, help='Output format (overrides config)')
@click.option('--strategies', default=None,
              help='Comma-separated strategy names (overrides config)')
@click.option('--system-prompt', default=None, help='System prompt for OpenAI format')
def generate(config_path, output, fmt, strategies, system_prompt):
    """Generate fine-tuning Q&A pairs from the Knowledge Graph."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    from .finetune.generator import FinetuneGenerator
    from .finetune.filters import FilterConfig

    # Load config if provided
    if config_path:
        config = load_config(config_path)
        ftc = config.finetune
    else:
        ftc = FinetuneConfig()

    # CLI overrides
    format_name = fmt or ftc.format
    strategy_names = strategies.split(",") if strategies else ftc.strategies
    prompt = system_prompt or ftc.system_prompt

    filter_config = FilterConfig(
        min_answer_length=ftc.min_answer_length,
        max_answer_length=ftc.max_answer_length,
        min_question_length=ftc.min_question_length,
        deduplicate=ftc.deduplicate,
    )

    output_dir = Path(output)

    print("\n" + "=" * 60)
    print("  Fine-Tuning Pair Generator")
    print("=" * 60)
    print(f"  KG Directory:  {output_dir}")
    print(f"  Format:        {format_name}")
    print(f"  Strategies:    {', '.join(strategy_names)}")
    print("=" * 60 + "\n")

    generator = FinetuneGenerator(
        kg_dir=output_dir,
        output_dir=output_dir,
        strategy_names=strategy_names,
        format_name=format_name,
        system_prompt=prompt,
        filter_config=filter_config,
    )

    report = generator.run()

    print(f"\n  Summary:")
    print(f"    Generated: {report.total_generated}")
    print(f"    Accepted:  {report.total_accepted}")
    print(f"    Rejected:  {report.total_rejected}")
    for name, stats in report.per_strategy.items():
        print(f"    [{name}] {stats['accepted']}/{stats['generated']} ({stats['acceptance_rate']}%)")
    print("=" * 60)


def main():
    cli()


if __name__ == "__main__":
    main()
