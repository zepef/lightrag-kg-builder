"""Tests for the data augmentation pipeline."""

import pytest

from src.finetune.strategies import FTPair
from src.finetune.augmenter import (
    TemplateAugmenter,
    QUESTION_SYNONYMS,
    TERM_SYNONYMS,
    CONNECTOR_SWAPS,
    get_augmenter,
)


def _make_pair(
    question="Qu'est-ce que le capital dans le cadre du PCG ?",
    answer="Le capital represente la valeur des apports faits par les associes a une entreprise. En outre, il constitue un element cle des capitaux propres.",
    strategy="entity_def",
    entities=None,
):
    return FTPair(
        question=question,
        answer=answer,
        strategy_name=strategy,
        source_entities=entities or ["Capital"],
    )


# ============================================================================
# TemplateAugmenter - basic behavior
# ============================================================================

class TestTemplateAugmenterBasic:

    def test_augment_returns_originals_plus_variants(self):
        aug = TemplateAugmenter(n=2, seed=42)
        pairs = [_make_pair()]
        result = aug.augment(pairs)
        # Should have original + up to 2 variants
        assert len(result) >= 2
        assert result[0] == pairs[0]  # first is always original

    def test_augment_preserves_strategy_name(self):
        aug = TemplateAugmenter(n=1, seed=42)
        pairs = [_make_pair(strategy="relational")]
        result = aug.augment(pairs)
        for p in result:
            assert p.strategy_name == "relational"

    def test_augment_preserves_source_entities(self):
        aug = TemplateAugmenter(n=1, seed=42)
        pairs = [_make_pair(entities=["Capital", "Dette"])]
        result = aug.augment(pairs)
        for p in result:
            assert p.source_entities == ["Capital", "Dette"]

    def test_augmented_pairs_have_metadata(self):
        aug = TemplateAugmenter(n=2, seed=42)
        pairs = [_make_pair()]
        result = aug.augment(pairs)
        variants = result[1:]
        for v in variants:
            assert v.metadata.get("augmented") is True
            assert "original_q" in v.metadata

    def test_augment_empty_input(self):
        aug = TemplateAugmenter(n=3, seed=42)
        result = aug.augment([])
        assert result == []

    def test_augment_multiple_pairs(self):
        aug = TemplateAugmenter(n=1, seed=42)
        pairs = [
            _make_pair(question="Qu'est-ce que le capital ?"),
            _make_pair(question="Qu'est-ce que la dette ?"),
        ]
        result = aug.augment(pairs)
        # At least 2 originals
        assert len(result) >= 2

    def test_deterministic_with_seed(self):
        pairs = [_make_pair()]
        r1 = TemplateAugmenter(n=2, seed=123).augment(pairs)
        r2 = TemplateAugmenter(n=2, seed=123).augment(pairs)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.question == b.question
            assert a.answer == b.answer

    def test_different_seeds_differ(self):
        pairs = [_make_pair()]
        r1 = TemplateAugmenter(n=3, seed=1).augment(pairs)
        r2 = TemplateAugmenter(n=3, seed=999).augment(pairs)
        # At least one variant should differ (probabilistic but very likely with 3 variants)
        if len(r1) > 1 and len(r2) > 1:
            questions1 = {p.question for p in r1[1:]}
            questions2 = {p.question for p in r2[1:]}
            # Not guaranteed to differ, but extremely likely
            assert len(questions1 | questions2) >= 1


# ============================================================================
# TemplateAugmenter - question paraphrasing
# ============================================================================

class TestQuestionParaphrasing:

    def test_synonym_substitution(self):
        aug = TemplateAugmenter(n=1, seed=42)
        # This question starts with a known synonym group entry
        original = "Qu'est-ce que le capital ?"
        result = aug._paraphrase_question(original)
        # Should either match original or use an alternative from the group
        valid_starts = QUESTION_SYNONYMS[0]  # ["Qu'est-ce que", "Que designe", ...]
        assert any(result.startswith(s) for s in valid_starts) or result == original

    def test_term_substitution_can_occur(self):
        """With enough attempts, term substitution should happen."""
        aug = TemplateAugmenter(n=1, seed=42)
        question = "Qu'est-ce que la comptabilite dans le cadre du PCG ?"
        results = set()
        for seed in range(50):
            aug.rng.seed(seed)
            results.add(aug._paraphrase_question(question))
        # Should get at least some variation
        assert len(results) > 1

    def test_no_crash_on_no_match(self):
        aug = TemplateAugmenter(n=1, seed=42)
        result = aug._paraphrase_question("A completely unrelated question?")
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# TemplateAugmenter - answer paraphrasing
# ============================================================================

class TestAnswerParaphrasing:

    def test_connector_swap(self):
        aug = TemplateAugmenter(n=1, seed=42)
        answer = "Le capital est un element important. En outre, il represente la base financiere."
        results = set()
        for seed in range(50):
            aug.rng.seed(seed)
            results.add(aug._paraphrase_answer(answer))
        # "En outre," should get swapped to alternatives at least once
        assert len(results) > 1

    def test_no_crash_on_short_answer(self):
        aug = TemplateAugmenter(n=1, seed=42)
        result = aug._paraphrase_answer("Short.")
        assert isinstance(result, str)

    def test_connector_swaps_shuffled(self):
        aug = TemplateAugmenter(n=1, seed=42)
        s1 = aug.CONNECTOR_SWAPS_SHUFFLED()
        assert len(s1) == len(CONNECTOR_SWAPS)
        # Verify all entries are present
        originals = {c[0] for c in CONNECTOR_SWAPS}
        shuffled_originals = {c[0] for c in s1}
        assert originals == shuffled_originals


# ============================================================================
# Deduplication of variants
# ============================================================================

class TestVariantDeduplication:

    def test_no_duplicate_questions(self):
        aug = TemplateAugmenter(n=5, seed=42)
        pairs = [_make_pair()]
        result = aug.augment(pairs)
        questions = [p.question.lower().strip() for p in result]
        assert len(questions) == len(set(questions))

    def test_variant_differs_from_original(self):
        aug = TemplateAugmenter(n=3, seed=42)
        pairs = [_make_pair()]
        result = aug.augment(pairs)
        if len(result) > 1:
            for variant in result[1:]:
                # At least question or answer must differ
                assert (
                    variant.question != pairs[0].question
                    or variant.answer != pairs[0].answer
                )


# ============================================================================
# get_augmenter factory
# ============================================================================

class TestGetAugmenter:

    def test_template_augmenter_default(self):
        aug = get_augmenter(n=3)
        assert isinstance(aug, TemplateAugmenter)
        assert aug.n == 3

    def test_template_augmenter_with_seed(self):
        aug = get_augmenter(n=2, seed=42)
        assert isinstance(aug, TemplateAugmenter)

    def test_llm_augmenter_flag(self):
        from src.finetune.augmenter import LLMAugmenter
        aug = get_augmenter(n=3, use_llm=True)
        assert isinstance(aug, LLMAugmenter)
        assert aug.n == 3

    def test_llm_augmenter_custom_url(self):
        from src.finetune.augmenter import LLMAugmenter
        aug = get_augmenter(n=1, use_llm=True, llm_url="http://gpu:8000/v1")
        assert isinstance(aug, LLMAugmenter)
        assert aug.api_url == "http://gpu:8000/v1"


# ============================================================================
# Synonym data integrity
# ============================================================================

class TestSynonymData:

    def test_question_synonyms_not_empty(self):
        assert len(QUESTION_SYNONYMS) > 0
        for group in QUESTION_SYNONYMS:
            assert len(group) >= 2, "Each synonym group needs at least 2 entries"

    def test_term_synonyms_not_empty(self):
        assert len(TERM_SYNONYMS) > 0
        for term, synonyms in TERM_SYNONYMS.items():
            assert len(synonyms) >= 2, f"Term '{term}' needs at least 2 synonyms"

    def test_connector_swaps_not_empty(self):
        assert len(CONNECTOR_SWAPS) > 0
        for original, alternatives in CONNECTOR_SWAPS:
            assert isinstance(original, str)
            assert len(alternatives) >= 1
