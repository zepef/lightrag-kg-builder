"""
Output formatters for fine-tuning pairs.

Supports OpenAI, Alpaca, and ShareGPT formats.
"""

import json
import logging
from pathlib import Path
from typing import List

from .strategies import FTPair

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Tu es un expert-comptable specialise dans le Plan Comptable General (PCG) 2026. "
    "Tu reponds de maniere precise et pedagogique aux questions de comptabilite francaise."
)


def format_openai(pair: FTPair, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict:
    """Format as OpenAI fine-tuning JSONL."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.answer},
        ]
    }


def format_alpaca(pair: FTPair, **_kwargs) -> dict:
    """Format as Alpaca-style JSONL."""
    return {
        "instruction": pair.question,
        "input": "",
        "output": pair.answer,
    }


def format_sharegpt(pair: FTPair, **_kwargs) -> dict:
    """Format as ShareGPT-style JSONL."""
    return {
        "conversations": [
            {"from": "human", "value": pair.question},
            {"from": "gpt", "value": pair.answer},
        ]
    }


FORMATTERS = {
    "openai": format_openai,
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
}


def write_jsonl(
    pairs: List[FTPair],
    output_path: Path,
    format_name: str = "openai",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> int:
    """
    Write pairs to a JSONL file in the specified format.

    Args:
        pairs: List of FTPairs to write
        output_path: Path to output JSONL file
        format_name: One of "openai", "alpaca", "sharegpt"
        system_prompt: System prompt for OpenAI format

    Returns:
        Number of pairs written
    """
    formatter = FORMATTERS.get(format_name)
    if formatter is None:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(FORMATTERS.keys())}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = formatter(pair, system_prompt=system_prompt)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Wrote {count} pairs to {output_path} ({format_name} format)")
    return count


def write_rejected(
    rejected: list,
    output_path: Path,
) -> int:
    """Write rejected pairs with reasons for review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rejected:
            record = {
                "question": r.pair.question,
                "answer": r.pair.answer,
                "strategy": r.pair.strategy_name,
                "reason": r.reason,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Wrote {count} rejected pairs to {output_path}")
    return count
