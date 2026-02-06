# lightrag-kg-builder

Reusable Knowledge Graph build pipeline using LightRAG with parallel vLLM processing.

Designed for structured legal documents (accounting standards, civil code, tax code) but extensible to any domain through the **chunking profile** system.

## Quick Start

```bash
# Clone
git clone https://github.com/zepef/lightrag-kg-builder.git
cd lightrag-kg-builder

# Install dependencies
pip install -r requirements.txt

# Check services are running
./scripts/health_check.sh

# Build a Knowledge Graph
python -m src.cli build \
  --config configs/pcg2026.yaml \
  --sources /path/to/pdfs \
  --output /path/to/kg \
  --mode 4090
```

## Prerequisites

**1. vLLM server** (for LLM inference):
```bash
# RTX 4090 (24GB)
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6

# 2x RTX 4090 (48GB)
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85
```

**2. Ollama** (for embeddings):
```bash
ollama pull nomic-embed-text
ollama serve
```

## CLI Usage

### Build

```bash
# Build with config file
python -m src.cli build --config configs/pcg2026.yaml --sources /path/to/pdfs --output /path/to/kg

# Hardware presets
python -m src.cli build --config configs/pcg2026.yaml --sources ./sources --output ./kg --mode dual4090
python -m src.cli build --config configs/pcg2026.yaml --sources ./sources --output ./kg --mode 8x5090

# Override pipeline count
python -m src.cli build --config configs/pcg2026.yaml --sources ./sources --output ./kg --pipelines 8

# Test with small chunk count
python -m src.cli build --config configs/pcg2026.yaml --sources ./sources --output ./kg --test-chunks 5

# Clean start
python -m src.cli build --config configs/pcg2026.yaml --sources ./sources --output ./kg --clean
```

### Status

```bash
python -m src.cli status --output /path/to/kg
```

### Health Check

```bash
python -m src.cli health --vllm-url http://localhost:8000/v1 --ollama-url http://localhost:11434
```

## Hardware Presets

| Preset | GPUs | Pipelines | Estimated Speedup |
|--------|------|-----------|-------------------|
| `4090` | 1x RTX 4090 | 2 | ~2x |
| `dual4090` | 2x RTX 4090 | 4 | ~4x |
| `8x5090` | 8x RTX 5090 | 16 | ~12x |
| `bigpu` | 8x A100/H100 | 4 | ~4x |
| `max` | 8x H100 NVLink | 16 | ~10x |

## Configuration

Config files use YAML format. See `configs/pcg2026.yaml` for a complete example.

```yaml
project:
  name: "PCG 2026"

chunking:
  profile: pcg              # Chunking profile name
  max_chunk_size: 4000
  min_chunk_size: 500

llm:
  provider: vllm
  vllm_url: http://localhost:8000/v1
  ollama_url: http://localhost:11434
  model: mistralai/Mistral-7B-Instruct-v0.3
  embedding_model: nomic-embed-text
  embedding_dim: 768
```

## Chunking Profiles

The profile system lets you define domain-specific document patterns. Each profile specifies:

- **patterns**: Regex patterns for structural elements (titles, chapters, articles, etc.)
- **hierarchy**: Ordered list of hierarchy levels
- **atomic_units**: Elements that should never be split across chunks
- **context_format**: Template for context prefixes

Built-in profiles:
- `pcg` — Plan Comptable General (French accounting standard)

To add a new profile, create a file in `src/chunkers/profiles/` and register it in `src/cli.py`'s `get_chunker()` function.

## Project Structure

```
lightrag-kg-builder/
├── configs/                    # YAML config files per project
├── src/
│   ├── cli.py                  # Click CLI entry point
│   ├── extractors/             # PDF text extraction
│   ├── chunkers/               # Document chunking with profile system
│   │   ├── base.py             # ChunkingProfile dataclass
│   │   ├── legal_chunker.py    # Generic legal document chunker
│   │   └── profiles/           # Domain-specific patterns
│   ├── kg/                     # Knowledge graph building
│   │   ├── builder.py          # Core LightRAG builder (Ollama)
│   │   ├── parallel.py         # Multiprocessing orchestration
│   │   └── merger.py           # Graph merging and deduplication
│   ├── llm/                    # LLM client (vLLM)
│   └── utils/                  # Journal, config loader
├── scripts/                    # Shell utilities
└── docs/                       # Architecture documentation
```

## Integration as Submodule

To use in another project (e.g., parles-au-pcg):

```bash
git submodule add https://github.com/zepef/lightrag-kg-builder.git kg-builder
```

Then reference `kg-builder/src/cli.py` in your build scripts.

## Output

After a successful build, the output directory contains:

- `merged/graph_chunk_entity_relation.graphml` — The merged knowledge graph (open with Gephi or NetworkX)
- `p1/`, `p2/`, ... — Individual pipeline outputs
- `extracted_text.txt` — Cached PDF extraction
- `chunks_cache.json` — Cached semantic chunks
- `timing_report.json` — Build performance metrics
