"""Tests for chunking profiles and the LegalDocumentChunker."""

import pytest

from src.chunkers.base import ChunkingProfile, ChunkResult
from src.chunkers.legal_chunker import LegalDocumentChunker
from src.chunkers.profiles.pcg import PCG_PROFILE


# ============================================================================
# ChunkingProfile
# ============================================================================

class TestChunkingProfile:

    def test_default_split_on_uses_hierarchy(self):
        profile = ChunkingProfile(
            name="test",
            patterns={"a": r"^A (\d+) (.+)$", "b": r"^B (\d+) (.+)$", "c": r"^C (\d+) (.+)$"},
            hierarchy=["a", "b", "c"],
            atomic_units=[],
            context_format="[{a} > {b}]",
        )
        assert profile.split_on == ["b", "c"]

    def test_explicit_split_on_overrides_default(self):
        profile = ChunkingProfile(
            name="test",
            patterns={"a": r"^A$", "b": r"^B$"},
            hierarchy=["a", "b"],
            atomic_units=[],
            context_format="",
            split_on=["a"],
        )
        assert profile.split_on == ["a"]

    def test_single_level_hierarchy_split_on(self):
        profile = ChunkingProfile(
            name="test",
            patterns={"only": r"^Only (.+)$"},
            hierarchy=["only"],
            atomic_units=[],
            context_format="",
        )
        assert profile.split_on == ["only"]

    def test_compiled_patterns_match(self):
        profile = ChunkingProfile(
            name="test",
            patterns={"level": r"^Section (\d+) - (.+)$"},
            hierarchy=["level"],
            atomic_units=[],
            context_format="",
        )
        compiled = profile.get_compiled_patterns()
        assert "level" in compiled
        match = compiled["level"].search("Section 1 - Introduction")
        assert match is not None
        assert match.group(1) == "1"
        assert match.group(2) == "Introduction"

    def test_compiled_patterns_no_match(self):
        profile = ChunkingProfile(
            name="test",
            patterns={"level": r"^Section (\d+) - (.+)$"},
            hierarchy=["level"],
            atomic_units=[],
            context_format="",
        )
        compiled = profile.get_compiled_patterns()
        match = compiled["level"].search("This does not match")
        assert match is None


# ============================================================================
# PCG Profile Patterns
# ============================================================================

class TestPCGProfile:

    def test_livre_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["livre"].search("Livre I : Principes generaux")
        assert match is not None
        assert match.group(1) == "I"

    def test_livre_pattern_with_digit(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["livre"].search("Livre 2 - Dispositions")
        assert match is not None
        assert match.group(1) == "2"

    def test_titre_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["titre"].search("TITRE II \u2013 Modalites")
        assert match is not None
        assert match.group(1) == "II"

    def test_chapitre_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["chapitre"].search("CHAPITRE III \u2013 Regles d'evaluation")
        assert match is not None
        assert match.group(1) == "III"

    def test_section_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["section"].search("Section 1 - Principes")
        assert match is not None
        assert match.group(1) == "1"

    def test_article_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["article"].search("Article 211-1 : Definition des actifs")
        assert match is not None
        assert match.group(1) == "211-1"

    def test_article_pattern_multi_segment(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["article"].search("Article 512-1-1 : Cas special")
        assert match is not None
        assert match.group(1) == "512-1-1"

    def test_account_def_pattern(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["account_def"].search("101 Capital social")
        assert match is not None
        assert match.group(1) == "101"
        assert match.group(2) == "Capital social"

    def test_account_def_5_digits(self):
        compiled = PCG_PROFILE.get_compiled_patterns()
        match = compiled["account_def"].search("60110 Achats de matieres premieres")
        assert match is not None
        assert match.group(1) == "60110"

    def test_hierarchy_order(self):
        assert PCG_PROFILE.hierarchy == ["livre", "titre", "chapitre", "section", "sous_section"]

    def test_atomic_units(self):
        assert "article" in PCG_PROFILE.atomic_units
        assert "account_def" in PCG_PROFILE.atomic_units

    def test_split_on(self):
        assert PCG_PROFILE.split_on == ["titre", "chapitre", "section"]


# ============================================================================
# ChunkResult
# ============================================================================

class TestChunkResult:

    @pytest.fixture
    def simple_profile(self):
        return ChunkingProfile(
            name="test",
            patterns={},
            hierarchy=["part", "chapter"],
            atomic_units=[],
            context_format="[{part} > {chapter}]",
        )

    def test_context_prefix(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Some content",
            context={"part": "Part 1", "chapter": "Chapter 2"},
            char_count=12,
        )
        prefix = chunk.get_context_prefix(simple_profile)
        assert "Part: Part 1" in prefix
        assert "Chapter: Chapter 2" in prefix
        assert " > " in prefix

    def test_context_prefix_partial(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Content",
            context={"part": "Part 1", "chapter": None},
            char_count=7,
        )
        prefix = chunk.get_context_prefix(simple_profile)
        assert "Part 1" in prefix
        assert "Chapter" not in prefix

    def test_context_prefix_empty(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Content",
            context={"part": None, "chapter": None},
            char_count=7,
        )
        prefix = chunk.get_context_prefix(simple_profile)
        assert prefix == ""

    def test_to_text_with_context(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Body text here",
            context={"part": "P1", "chapter": "C1"},
            char_count=14,
        )
        text = chunk.to_text(simple_profile, include_context=True)
        assert "Body text here" in text
        assert "[" in text

    def test_to_text_without_context(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Body text here",
            context={"part": "P1", "chapter": "C1"},
            char_count=14,
        )
        text = chunk.to_text(simple_profile, include_context=False)
        assert text == "Body text here"

    def test_to_dict_keys(self, simple_profile):
        chunk = ChunkResult(
            chunk_id="test-0001",
            chunk_type="section",
            content="Body",
            context={"part": "P1", "chapter": None},
            atomic_ids=["art-1"],
            char_count=4,
        )
        d = chunk.to_dict(simple_profile)
        assert d["id"] == "test-0001"
        assert "Body" in d["text"]
        assert d["metadata"]["type"] == "section"
        assert d["metadata"]["atomic_ids"] == ["art-1"]
        assert d["metadata"]["char_count"] == 4


# ============================================================================
# LegalDocumentChunker
# ============================================================================

SAMPLE_PCG_TEXT = """\
Livre I : Principes generaux

TITRE I \u2013 Objet et principes de la comptabilite

CHAPITRE I \u2013 Champ d'application

Article 111-1 : Les dispositions du present reglement s'appliquent a toute personne \
physique ou morale soumise a l'obligation legale d'etablir des comptes annuels comprenant \
le bilan, le compte de resultat et une annexe. Ces comptes annuels sont conformes aux \
prescriptions comptables du present reglement.

Article 111-2 : La comptabilite est un systeme d'organisation de l'information financiere \
permettant de saisir, classer, enregistrer des donnees de base chiffrees et presenter des \
etats refletant une image fidele du patrimoine, de la situation financiere et du resultat \
de l'entite a la date de cloture.

CHAPITRE II \u2013 Principes

Article 112-1 : La comptabilite est conforme aux regles et procedures en vigueur qui sont \
appliquees avec sincerite afin de traduire la connaissance que les responsables de \
l'etablissement des comptes ont de la realite et de l'importance relative des evenements \
enregistres. Elle respecte le principe de prudence.

Article 112-2 : La presentation des comptes annuels est fondee sur le principe de \
continuite d'exploitation. Les methodes comptables adoptees doivent etre appliquees de \
maniere coherente d'un exercice a l'autre."""


class TestLegalDocumentChunker:

    def test_parse_structure_finds_elements(self):
        chunker = LegalDocumentChunker(PCG_PROFILE)
        elements = chunker.parse_structure(SAMPLE_PCG_TEXT)
        types = [e["type"] for e in elements]
        assert "livre" in types
        assert "titre" in types
        assert "chapitre" in types
        assert "article" in types

    def test_parse_structure_order(self):
        chunker = LegalDocumentChunker(PCG_PROFILE)
        elements = chunker.parse_structure(SAMPLE_PCG_TEXT)
        positions = [e["start"] for e in elements]
        assert positions == sorted(positions), "Elements should be sorted by position"

    def test_extract_sections(self):
        chunker = LegalDocumentChunker(PCG_PROFILE)
        sections = chunker.extract_sections(SAMPLE_PCG_TEXT)
        assert len(sections) > 0
        for context, elem, content in sections:
            assert isinstance(context, dict)
            assert isinstance(content, str)
            assert len(content) > 0

    def test_chunk_document_returns_dicts(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=4000, min_chunk_size=100)
        chunks = chunker.chunk_document(SAMPLE_PCG_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk

    def test_chunk_ids_are_unique(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=4000, min_chunk_size=100)
        chunks = chunker.chunk_document(SAMPLE_PCG_TEXT)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunk_ids_follow_profile_prefix(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=4000, min_chunk_size=100)
        chunks = chunker.chunk_document(SAMPLE_PCG_TEXT)
        for chunk in chunks:
            assert chunk["id"].startswith("pcg-")

    def test_chunk_respects_max_size(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=500, min_chunk_size=50)
        chunks = chunker.create_chunks(SAMPLE_PCG_TEXT)
        for chunk in chunks:
            # Chunks exceeding 1.5x max_size get split in post-processing
            assert chunk.char_count <= 500 * 1.5

    def test_min_chunk_size_filters_small_sections(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=4000, min_chunk_size=10000)
        chunks = chunker.create_chunks(SAMPLE_PCG_TEXT)
        # With min_chunk_size larger than any section, nothing should be produced
        assert len(chunks) == 0

    def test_chunk_context_includes_hierarchy(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=4000, min_chunk_size=100)
        chunks = chunker.chunk_document(SAMPLE_PCG_TEXT)
        has_bracket_context = any("[" in c["text"] for c in chunks)
        assert has_bracket_context

    def test_no_context_when_disabled(self):
        chunker = LegalDocumentChunker(
            PCG_PROFILE, max_chunk_size=4000, min_chunk_size=100, include_context=False
        )
        chunks = chunker.chunk_document(SAMPLE_PCG_TEXT)
        for chunk in chunks:
            assert not chunk["text"].startswith("[")

    def test_split_large_chunk(self):
        chunker = LegalDocumentChunker(PCG_PROFILE, max_chunk_size=200, min_chunk_size=50)
        big_chunk = ChunkResult(
            chunk_id="test-0000",
            chunk_type="section",
            content="Paragraph one about accounting.\n\nParagraph two about reporting.\n\n"
                    "Paragraph three about compliance.\n\nParagraph four about standards.\n\n"
                    "Paragraph five about auditing.\n\nParagraph six about disclosure.",
            context={"livre": "I", "titre": "I", "chapitre": "I", "section": None, "sous_section": None},
            char_count=300,
        )
        # 300 > 200 * 1.5 = 300, so exactly at boundary. Adjust to ensure splitting.
        big_chunk.char_count = 400
        big_chunk.content = big_chunk.content + "\n\n" + "Extra long paragraph. " * 10
        sub_chunks = chunker._split_large_chunk(big_chunk, start_idx=0)
        assert len(sub_chunks) >= 2
        for sc in sub_chunks:
            assert sc.chunk_type == big_chunk.chunk_type
