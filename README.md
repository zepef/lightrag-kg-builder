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
│   ├── finetune/               # Fine-tuning pair generation
│   │   ├── loader.py           # KG data loader (GraphML + JSON stores)
│   │   ├── strategies.py       # 7 generation strategies
│   │   ├── filters.py          # Quality filtering pipeline
│   │   ├── formatter.py        # Output formatters (OpenAI/Alpaca/ShareGPT)
│   │   └── generator.py        # Orchestrator
│   ├── kg/                     # Knowledge graph building
│   │   ├── builder.py          # Core LightRAG builder (Ollama)
│   │   ├── parallel.py         # Multiprocessing orchestration
│   │   └── merger.py           # Graph merging and deduplication
│   ├── llm/                    # LLM client (vLLM)
│   └── utils/                  # Journal, config loader
├── scripts/                    # Shell utilities
│   └── visualize_kg.py         # Interactive graph visualization
└── docs/                       # Architecture documentation
```

## Fine-Tuning Pair Generation

Generate Q&A training pairs directly from the Knowledge Graph — no LLM required.

### Usage

```bash
# Generate with config (all 7 strategies, OpenAI format)
python -m src.cli generate --config configs/pcg2026.yaml --output /path/to/kg

# Select output format
python -m src.cli generate --output /path/to/kg --format alpaca

# Select specific strategies
python -m src.cli generate --output /path/to/kg --strategies entity_def,relational,chunk_qa
```

### Strategies

| # | Strategy | Source Data | Description |
|---|----------|-------------|-------------|
| 1 | `entity_def` | Entity descriptions | Definition-style Q&A from entity metadata |
| 2 | `relational` | Graph edges | Relationship questions between entities |
| 3 | `hierarchical` | Chunk context brackets | Section-based Q&A from document hierarchy |
| 4 | `comparative` | Same-type entity pairs | Compare/contrast entities of same type |
| 5 | `multihop` | 2-3 edge graph paths | Multi-hop reasoning chains |
| 6 | `chunk_qa` | Chunk text patterns | Heuristic extraction (definitions, articles, rules) |
| 7 | `thematic` | Entity type clusters | Grouped overview summaries |

### Output Formats

- **openai** — `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- **alpaca** — `{"instruction": "Q", "input": "", "output": "A"}`
- **sharegpt** — `{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}`

### Output Files

```
{output}/finetune/
├── pairs_openai.jsonl        # Generated training pairs
├── generation_report.json    # Stats per strategy, filter results
└── rejected_pairs.jsonl      # Filtered-out pairs for review
```

### Configuration

Add a `finetune` section to your config YAML:

```yaml
finetune:
  system_prompt: "Tu es un expert-comptable..."
  strategies: [entity_def, relational, hierarchical, comparative, multihop, chunk_qa, thematic]
  format: openai
  filters:
    min_answer_length: 50
    max_answer_length: 2000
    deduplicate: true
```

## Graph Visualization

Generate an interactive HTML visualization of the Knowledge Graph:

```bash
python scripts/visualize_kg.py --input /path/to/kg --output visualization.html
```

Features: dark theme, color-coded entity types, node size by degree centrality, hover tooltips, legend overlay.

If using as a submodule in parles-au-pcg, the `/kg-viz` Claude Code skill automates this.

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
