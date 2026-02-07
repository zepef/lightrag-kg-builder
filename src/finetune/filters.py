"""
Quality filtering pipeline for fine-tuning pairs.

Applies configurable quality checks and deduplication.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from .strategies import FTPair

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for quality filters."""
    min_answer_length: int = 50
    max_answer_length: int = 2000
    min_question_length: int = 10
    deduplicate: bool = True
    remove_empty_descriptions: bool = True


@dataclass
class RejectedPair:
    """A pair that was filtered out, with reason."""
    pair: FTPair
    reason: str


class QualityFilter:
    """
    Quality filtering pipeline for FTPairs.

    Returns (accepted, rejected) tuple with rejection reasons.
    """

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()

    def filter(self, pairs: List[FTPair]) -> Tuple[List[FTPair], List[RejectedPair]]:
        """
        Apply all quality filters to a list of pairs.

        Returns:
            Tuple of (accepted pairs, rejected pairs with reasons)
        """
        accepted = []
        rejected = []

        for pair in pairs:
            reason = self._check(pair)
            if reason:
                rejected.append(RejectedPair(pair=pair, reason=reason))
            else:
                accepted.append(pair)

        # Deduplication pass
        if self.config.deduplicate:
            accepted, dedup_rejected = self._deduplicate(accepted)
            rejected.extend(dedup_rejected)

        logger.info(
            f"Filter: {len(accepted)} accepted, {len(rejected)} rejected "
            f"(from {len(pairs)} input)"
        )
        return accepted, rejected

    def _check(self, pair: FTPair) -> str | None:
        """Check a single pair against quality rules. Returns rejection reason or None."""
        # Question length
        if len(pair.question.strip()) < self.config.min_question_length:
            return f"question_too_short ({len(pair.question)} < {self.config.min_question_length})"

        # Answer length
        answer_len = len(pair.answer.strip())
        if answer_len < self.config.min_answer_length:
            return f"answer_too_short ({answer_len} < {self.config.min_answer_length})"

        # Truncate overly long answers (not reject, but trim)
        if answer_len > self.config.max_answer_length:
            pair.answer = pair.answer[:self.config.max_answer_length].rsplit(".", 1)[0] + "."

        # Empty content
        if not pair.answer.strip() or not pair.question.strip():
            return "empty_content"

        # Remove pairs where the answer is just the entity name repeated
        if self.config.remove_empty_descriptions:
            for entity in pair.source_entities:
                if pair.answer.strip().lower() == entity.lower():
                    return "answer_is_entity_name"

        return None

    def _deduplicate(self, pairs: List[FTPair]) -> Tuple[List[FTPair], List[RejectedPair]]:
        """Remove near-duplicate pairs based on normalized question text."""
        seen = {}
        unique = []
        duplicates = []

        for pair in pairs:
            key = self._normalize(pair.question)
            if key in seen:
                duplicates.append(RejectedPair(pair=pair, reason=f"duplicate_of:{seen[key]}"))
            else:
                seen[key] = pair.question[:60]
                unique.append(pair)

        if duplicates:
            logger.info(f"Dedup: removed {len(duplicates)} duplicates")

        return unique, duplicates

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for dedup comparison."""
        import re
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t
