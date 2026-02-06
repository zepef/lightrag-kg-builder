# Architecture

## Pipeline Overview

```
PDFs → Extract → Chunk → Parallel KG Build → Merge → GraphML
```

### Phase 1: PDF Extraction
- `src/extractors/pdf_extractor.py`
- PyMuPDF-based extraction with configurable cleanup patterns
- Output: `extracted_text.txt` (cached for re-runs)

### Phase 2: Semantic Chunking
- `src/chunkers/legal_chunker.py` + `src/chunkers/profiles/`
- Profile-driven: regex patterns define document structure
- Respects hierarchy (Livre > Titre > Chapitre > Section)
- Keeps atomic units together (articles, account definitions)
- Output: `chunks_cache.json` (cached for re-runs)

### Phase 3: Parallel KG Build
- `src/kg/parallel.py`
- N parallel OS processes, each with its own LightRAG instance
- Shared vLLM server handles continuous batching across all pipelines
- Shared Ollama server handles embeddings
- Each pipeline writes to its own directory (`p1/`, `p2/`, ...)

```
+-------------------------------------------+
|  vLLM Server (continuous batching)        |
+-------------------------------------------+
       ^           ^           ^
+--------+ +--------+ +--------+
| Pipe 1 | | Pipe 2 | | Pipe N |
| kg/p1/ | | kg/p2/ | | kg/pN/ |
+--------+ +--------+ +--------+
       |           |           |
       +-----+-----+---------+
             v
+-------------------------------------------+
|  Graph Merger (entity deduplication)      |
+-------------------------------------------+
```

### Phase 4: Graph Merging
- `src/kg/merger.py`
- Loads GraphML from each pipeline
- Name-based entity deduplication
- Relationship consolidation (keeps longest description, max weight)
- Merges KV stores (full docs, text chunks, LLM cache)
- Output: `merged/graph_chunk_entity_relation.graphml`

## Chunking Profile System

Profiles are dataclasses that define how to parse a specific document type:

```python
@dataclass
class ChunkingProfile:
    name: str                           # Profile identifier
    patterns: Dict[str, str]            # level_name -> regex
    hierarchy: List[str]                # ordered levels
    atomic_units: List[str]             # never split these
    context_format: str                 # context prefix template
    split_on: Optional[List[str]]       # levels that force new chunks
```

The `LegalDocumentChunker` is generic — it takes any profile and produces
chunks that respect the defined hierarchy and atomic units.

### Adding a New Profile

1. Create `src/chunkers/profiles/your_domain.py`:
```python
from ..base import ChunkingProfile

YOUR_PROFILE = ChunkingProfile(
    name="your_domain",
    patterns={
        'part': r'^Part\s+(\d+)\s*[:\-]\s*(.+)$',
        'chapter': r'^Chapter\s+(\d+)\s*[:\-]\s*(.+)$',
        'section': r'^Section\s+(\d+)\s*[:\-]\s*(.+)$',
        'article': r'^Article\s+(\d+)\s*[:\-]?\s*(.*)$',
    },
    hierarchy=['part', 'chapter', 'section'],
    atomic_units=['article'],
    context_format="[{part} > {chapter} > {section}]",
)
```

2. Register in `src/cli.py`:
```python
def get_chunker(profile_name, ...):
    if profile_name == "your_domain":
        from .chunkers.profiles.your_domain import YOUR_PROFILE
        return LegalDocumentChunker(YOUR_PROFILE, ...)
```

3. Create a config file in `configs/your_domain.yaml`.

## LLM Integration

Two LLM backends are supported:

- **vLLM** (`src/llm/vllm_client.py`): OpenAI-compatible API, used for KG entity extraction via parallel pipelines. Provides continuous batching for high throughput.
- **Ollama** (via LightRAG): Used for embeddings (nomic-embed-text). Can also serve as LLM backend via `src/kg/builder.py` for single-pipeline builds.

## Crash Recovery

- **Chunk-level checkpointing**: Each pipeline tracks completed chunk IDs in `progress.json`
- **Cached extraction**: PDF extraction and chunking are cached; re-runs skip completed phases
- **Execution journal** (`src/utils/journal.py`): Append-only JSONL log for debugging and resume state detection

## Key Design Decisions

1. **Multiprocessing over async**: Each pipeline runs in its own OS process with its own event loop. This prevents LightRAG's internal async operations from blocking other pipelines.

2. **Profile-based chunking**: Document structure patterns are data, not code. Adding a new document type means creating a profile dataclass, not writing a new chunker.

3. **Config files over CLI args**: YAML configs capture project-specific settings (model, chunk size, cleanup patterns). CLI args override for quick experimentation.
