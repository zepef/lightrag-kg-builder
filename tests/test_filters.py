"""Tests for the quality filtering pipeline."""

import pytest

from src.finetune.strategies import FTPair
from src.finetune.filters import QualityFilter, FilterConfig, RejectedPair


def _make_pair(
    question="Qu'est-ce que le capital?",
    answer="Le capital represente la valeur des apports faits par les associes a une entreprise.",
    strategy="test",
    entities=None,
):
    return FTPair(
        question=question,
        answer=answer,
        strategy_name=strategy,
        source_entities=entities or [],
    )


# ============================================================================
# Basic acceptance / rejection
# ============================================================================

class TestQualityFilterBasic:

    def test_accept_valid_pair(self):
        config = FilterConfig(min_answer_length=10, min_question_length=5)
        f = QualityFilter(config)
        accepted, rejected = f.filter([_make_pair()])
        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_reject_short_answer(self):
        config = FilterConfig(min_answer_length=100)
        f = QualityFilter(config)
        pair = _make_pair(answer="Too short")
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "answer_too_short" in rejected[0].reason

    def test_reject_short_question(self):
        config = FilterConfig(min_question_length=50)
        f = QualityFilter(config)
        pair = _make_pair(question="Q?")
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0
        assert "question_too_short" in rejected[0].reason

    def test_reject_empty_question(self):
        config = FilterConfig(min_answer_length=1, min_question_length=1)
        f = QualityFilter(config)
        pair = _make_pair(question="   ", answer="A valid answer here.")
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0

    def test_reject_empty_answer(self):
        config = FilterConfig(min_answer_length=1, min_question_length=1)
        f = QualityFilter(config)
        pair = _make_pair(question="A valid question?", answer="   ")
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0


# ============================================================================
# Entity name / empty description filter
# ============================================================================

class TestEntityNameFilter:

    def test_reject_answer_is_entity_name(self):
        config = FilterConfig(min_answer_length=1, remove_empty_descriptions=True)
        f = QualityFilter(config)
        pair = _make_pair(answer="Capital", entities=["Capital"])
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0
        assert "answer_is_entity_name" in rejected[0].reason

    def test_case_insensitive_entity_match(self):
        config = FilterConfig(min_answer_length=1, remove_empty_descriptions=True)
        f = QualityFilter(config)
        pair = _make_pair(answer="capital", entities=["Capital"])
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 0

    def test_accept_when_answer_differs_from_entity(self):
        config = FilterConfig(min_answer_length=1, remove_empty_descriptions=True)
        f = QualityFilter(config)
        pair = _make_pair(answer="Capital is the equity contribution.", entities=["Capital"])
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 1

    def test_skip_entity_check_when_disabled(self):
        config = FilterConfig(min_answer_length=1, remove_empty_descriptions=False)
        f = QualityFilter(config)
        pair = _make_pair(answer="Capital", entities=["Capital"])
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 1


# ============================================================================
# Answer truncation
# ============================================================================

class TestTruncation:

    def test_truncate_long_answer(self):
        config = FilterConfig(max_answer_length=50, min_answer_length=10)
        f = QualityFilter(config)
        long_answer = "This is a sentence. " * 10  # ~200 chars
        pair = _make_pair(answer=long_answer)
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 1
        assert len(accepted[0].answer) < len(long_answer)

    def test_no_truncation_when_within_limit(self):
        config = FilterConfig(max_answer_length=500, min_answer_length=10)
        f = QualityFilter(config)
        answer = "A normal length answer that fits."
        pair = _make_pair(answer=answer)
        accepted, rejected = f.filter([pair])
        assert len(accepted) == 1
        assert accepted[0].answer == answer


# ============================================================================
# Deduplication
# ============================================================================

class TestDeduplication:

    def test_dedup_exact_match(self):
        config = FilterConfig(min_answer_length=10, deduplicate=True)
        f = QualityFilter(config)
        pairs = [
            _make_pair(question="What is capital?", answer="Capital is equity contribution to a company."),
            _make_pair(question="What is capital?", answer="Capital represents ownership stake in entity."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert "duplicate" in rejected[0].reason

    def test_dedup_case_insensitive(self):
        config = FilterConfig(min_answer_length=10, deduplicate=True)
        f = QualityFilter(config)
        pairs = [
            _make_pair(question="What is capital?", answer="First answer about capital and accounting."),
            _make_pair(question="WHAT IS CAPITAL?", answer="Second answer about capital and accounting."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 1

    def test_dedup_ignores_punctuation(self):
        config = FilterConfig(min_answer_length=10, deduplicate=True)
        f = QualityFilter(config)
        pairs = [
            _make_pair(question="What is capital?", answer="Capital is equity contribution to a company."),
            _make_pair(question="What is capital", answer="Capital is something different and unique."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 1

    def test_no_dedup_when_disabled(self):
        config = FilterConfig(min_answer_length=10, deduplicate=False)
        f = QualityFilter(config)
        pairs = [
            _make_pair(question="What is capital?", answer="First answer about capital and accounting."),
            _make_pair(question="What is capital?", answer="Second answer about capital and accounting."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 2

    def test_dedup_keeps_first_occurrence(self):
        config = FilterConfig(min_answer_length=10, deduplicate=True)
        f = QualityFilter(config)
        first_answer = "The first and original answer about the topic."
        pairs = [
            _make_pair(question="Explain X.", answer=first_answer),
            _make_pair(question="Explain X!", answer="A different duplicate answer about X."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 1
        assert accepted[0].answer == first_answer


# ============================================================================
# Combined filters
# ============================================================================

class TestCombinedFilters:

    def test_multiple_rejection_reasons(self):
        config = FilterConfig(
            min_answer_length=20,
            min_question_length=10,
            deduplicate=True,
        )
        f = QualityFilter(config)
        pairs = [
            _make_pair(question="A valid long question?", answer="A valid long enough answer for this test."),
            _make_pair(question="Short", answer="A valid long enough answer for testing purposes."),
            _make_pair(question="Another valid question?", answer="Short"),
            _make_pair(question="A valid long question?", answer="Duplicate question, different answer text."),
        ]
        accepted, rejected = f.filter(pairs)
        assert len(accepted) == 1
        assert len(rejected) == 3

        reasons = [r.reason for r in rejected]
        assert any("question_too_short" in r for r in reasons)
        assert any("answer_too_short" in r for r in reasons)
        assert any("duplicate" in r for r in reasons)

    def test_empty_input(self):
        f = QualityFilter(FilterConfig())
        accepted, rejected = f.filter([])
        assert len(accepted) == 0
        assert len(rejected) == 0
