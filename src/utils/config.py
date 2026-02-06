"""
YAML Configuration Loader

Loads and validates project configuration from YAML files.
Supports merging config with CLI overrides.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class ProjectConfig:
    """Parsed and validated project configuration."""
    # Project
    project_name: str = "Untitled"

    # Chunking
    chunking_profile: str = "pcg"
    max_chunk_size: int = 4000
    min_chunk_size: int = 500

    # LLM
    llm_provider: str = "vllm"
    vllm_url: str = "http://localhost:8000/v1"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Paths (can be overridden by CLI)
    sources_dir: Optional[str] = None
    output_dir: Optional[str] = None

    # Extra cleanup patterns for PDF extraction
    cleanup_patterns: list = field(default_factory=list)


def load_config(config_path: str) -> ProjectConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated ProjectConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is malformed
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got: {type(raw)}")

    config = ProjectConfig()

    # Project section
    project = raw.get('project', {})
    if project.get('name'):
        config.project_name = project['name']

    # Chunking section
    chunking = raw.get('chunking', {})
    if chunking.get('profile'):
        config.chunking_profile = chunking['profile']
    if chunking.get('max_chunk_size'):
        config.max_chunk_size = int(chunking['max_chunk_size'])
    if chunking.get('min_chunk_size'):
        config.min_chunk_size = int(chunking['min_chunk_size'])

    # LLM section
    llm = raw.get('llm', {})
    if llm.get('provider'):
        config.llm_provider = llm['provider']
    if llm.get('vllm_url'):
        config.vllm_url = llm['vllm_url']
    if llm.get('ollama_url'):
        config.ollama_url = llm['ollama_url']
    if llm.get('model'):
        config.llm_model = llm['model']
    if llm.get('embedding_model'):
        config.embedding_model = llm['embedding_model']
    if llm.get('embedding_dim'):
        config.embedding_dim = int(llm['embedding_dim'])

    # Paths section
    paths = raw.get('paths', {})
    if paths.get('sources'):
        config.sources_dir = paths['sources']
    if paths.get('output'):
        config.output_dir = paths['output']

    # Extraction section
    extraction = raw.get('extraction', {})
    if extraction.get('cleanup_patterns'):
        config.cleanup_patterns = extraction['cleanup_patterns']

    return config


def merge_cli_overrides(config: ProjectConfig, **overrides) -> ProjectConfig:
    """
    Merge CLI argument overrides into config.

    Only non-None overrides are applied (CLI args that weren't specified
    remain at their config-file values).

    Args:
        config: Base config from YAML
        **overrides: CLI argument values (None = not specified)

    Returns:
        Updated ProjectConfig
    """
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config
