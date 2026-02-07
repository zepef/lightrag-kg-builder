"""
Fine-tuning pair generation orchestrator.

Chains: load -> generate -> filter -> format -> write
Produces output JSONL + generation report.
"""

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .loader import KGLoader
from .strategies import FTPair, get_strategies
from .filters import QualityFilter, FilterConfig
from .formatter import write_jsonl, write_rejected, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GenerationReport:
    """Statistics about the generation run."""
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    per_strategy: Dict[str, Dict] = None
    filter_config: Dict = None
    output_format: str = ""
    output_path: str = ""
    duration_ms: int = 0

    def __post_init__(self):
        if self.per_strategy is None:
            self.per_strategy = {}
        if self.filter_config is None:
            self.filter_config = {}

    def to_dict(self) -> dict:
        return {
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": round(
                self.total_accepted / max(self.total_generated, 1) * 100, 1
            ),
            "per_strategy": self.per_strategy,
            "filter_config": self.filter_config,
            "output_format": self.output_format,
            "output_path": self.output_path,
            "duration_ms": self.duration_ms,
        }


class FinetuneGenerator:
    """
    Orchestrates the full fine-tuning pair generation pipeline.

    Usage:
        generator = FinetuneGenerator(kg_dir="data/kg", output_dir="data/kg")
        report = generator.run()
    """

    def __init__(
        self,
        kg_dir: str | Path,
        output_dir: str | Path,
        strategy_names: Optional[List[str]] = None,
        format_name: str = "openai",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        filter_config: Optional[FilterConfig] = None,
    ):
        self.kg_dir = Path(kg_dir)
        self.output_dir = Path(output_dir) / "finetune"
        self.strategy_names = strategy_names
        self.format_name = format_name
        self.system_prompt = system_prompt
        self.filter_config = filter_config or FilterConfig()

    def run(self) -> GenerationReport:
        """Execute the full generation pipeline."""
        start = time.perf_counter()
        report = GenerationReport(
            filter_config={
                "min_answer_length": self.filter_config.min_answer_length,
                "max_answer_length": self.filter_config.max_answer_length,
                "min_question_length": self.filter_config.min_question_length,
                "deduplicate": self.filter_config.deduplicate,
            },
            output_format=self.format_name,
        )

        # Step 1: Load KG data
        print("  Loading KG data...")
        loader = KGLoader(self.kg_dir)
        loader.load()
        print(f"    {len(loader.entities)} entities, {len(loader.chunks)} chunks")

        # Step 2: Generate pairs from each strategy
        print("  Generating pairs...")
        strategies = get_strategies(self.strategy_names)
        all_pairs: List[FTPair] = []
        strategy_counts: Counter = Counter()

        for strategy in strategies:
            print(f"    [{strategy.name}] ...", end="", flush=True)
            pairs = strategy.generate(loader)
            all_pairs.extend(pairs)
            strategy_counts[strategy.name] = len(pairs)
            print(f" {len(pairs)} pairs")

        report.total_generated = len(all_pairs)
        print(f"    Total generated: {report.total_generated}")

        # Step 3: Filter
        print("  Filtering...")
        quality_filter = QualityFilter(self.filter_config)
        accepted, rejected = quality_filter.filter(all_pairs)
        report.total_accepted = len(accepted)
        report.total_rejected = len(rejected)
        print(f"    Accepted: {report.total_accepted}, Rejected: {report.total_rejected}")

        # Build per-strategy stats
        accepted_by_strategy = Counter(p.strategy_name for p in accepted)
        rejected_by_strategy = Counter(r.pair.strategy_name for r in rejected)
        rejection_reasons = Counter(r.reason.split("(")[0].strip() for r in rejected)

        for strategy_name in strategy_counts:
            generated = strategy_counts[strategy_name]
            acc = accepted_by_strategy.get(strategy_name, 0)
            rej = rejected_by_strategy.get(strategy_name, 0)
            report.per_strategy[strategy_name] = {
                "generated": generated,
                "accepted": acc,
                "rejected": rej,
                "acceptance_rate": round(acc / max(generated, 1) * 100, 1),
            }

        # Step 4: Format and write
        print(f"  Writing output ({self.format_name} format)...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.output_dir / f"pairs_{self.format_name}.jsonl"
        write_jsonl(accepted, output_path, self.format_name, self.system_prompt)
        report.output_path = str(output_path)

        # Write rejected for review
        rejected_path = self.output_dir / "rejected_pairs.jsonl"
        write_rejected(rejected, rejected_path)

        # Step 5: Write report
        report.duration_ms = int((time.perf_counter() - start) * 1000)

        report_dict = report.to_dict()
        report_dict["rejection_reasons"] = dict(rejection_reasons)

        report_path = self.output_dir / "generation_report.json"
        report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))

        print(f"\n  Generation complete in {report.duration_ms}ms")
        print(f"    Output: {output_path}")
        print(f"    Report: {report_path}")
        print(f"    Rejected: {rejected_path}")

        return report
