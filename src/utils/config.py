"""
YAML Configuration Loader

Loads and validates project configuration from YAML files.
Supports merging config with CLI overrides.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning pair generation."""
    system_prompt: str = (
        "Tu es un expert-comptable specialise dans le Plan Comptable General (PCG) 2026. "
        "Tu reponds de maniere precise et pedagogique aux questions de comptabilite francaise."
    )
    strategies: List[str] = field(default_factory=lambda: [
        "entity_def", "relational", "hierarchical", "comparative",
        "multihop", "chunk_qa", "thematic",
    ])
    format: str = "openai"
    min_answer_length: int = 50
    max_answer_length: int = 2000
    min_question_length: int = 10
    deduplicate: bool = True


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""
    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 50
    save_formats: List[str] = field(default_factory=lambda: ["lora", "gguf"])


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

    # Fine-tuning
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    # Training (LoRA)
    training: TrainingConfig = field(default_factory=TrainingConfig)


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

    # Finetune section
    ft = raw.get('finetune', {})
    if ft:
        ftc = config.finetune
        if ft.get('system_prompt'):
            ftc.system_prompt = ft['system_prompt']
        if ft.get('strategies'):
            ftc.strategies = ft['strategies']
        if ft.get('format'):
            ftc.format = ft['format']
        filters = ft.get('filters', {})
        if filters.get('min_answer_length') is not None:
            ftc.min_answer_length = int(filters['min_answer_length'])
        if filters.get('max_answer_length') is not None:
            ftc.max_answer_length = int(filters['max_answer_length'])
        if filters.get('deduplicate') is not None:
            ftc.deduplicate = bool(filters['deduplicate'])

    # Training section
    tr = raw.get('training', {})
    if tr:
        trc = config.training
        if tr.get('model'):
            trc.model = tr['model']
        if tr.get('max_seq_length') is not None:
            trc.max_seq_length = int(tr['max_seq_length'])
        if tr.get('load_in_4bit') is not None:
            trc.load_in_4bit = bool(tr['load_in_4bit'])
        if tr.get('lora_r') is not None:
            trc.lora_r = int(tr['lora_r'])
        if tr.get('lora_alpha') is not None:
            trc.lora_alpha = int(tr['lora_alpha'])
        if tr.get('lora_dropout') is not None:
            trc.lora_dropout = float(tr['lora_dropout'])
        if tr.get('epochs') is not None:
            trc.epochs = int(tr['epochs'])
        if tr.get('batch_size') is not None:
            trc.batch_size = int(tr['batch_size'])
        if tr.get('gradient_accumulation_steps') is not None:
            trc.gradient_accumulation_steps = int(tr['gradient_accumulation_steps'])
        if tr.get('learning_rate') is not None:
            trc.learning_rate = float(tr['learning_rate'])
        if tr.get('warmup_steps') is not None:
            trc.warmup_steps = int(tr['warmup_steps'])
        if tr.get('logging_steps') is not None:
            trc.logging_steps = int(tr['logging_steps'])
        if tr.get('save_steps') is not None:
            trc.save_steps = int(tr['save_steps'])
        if tr.get('save_formats'):
            trc.save_formats = tr['save_formats']

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
