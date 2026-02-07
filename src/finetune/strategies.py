"""
Fine-tuning pair generation strategies.

7 template-based strategies that exploit different aspects of the KG structure
to generate diverse Q&A training pairs. No LLM required.
"""

import logging
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .loader import KGLoader

logger = logging.getLogger(__name__)


@dataclass
class FTPair:
    """A single fine-tuning question-answer pair."""
    question: str
    answer: str
    strategy_name: str
    source_entities: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class Strategy(ABC):
    """Base class for generation strategies."""

    name: str = "base"

    @abstractmethod
    def generate(self, loader: KGLoader) -> List[FTPair]:
        """Generate Q&A pairs from the KG data."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1: Entity Definition
# ---------------------------------------------------------------------------

class EntityDefinitionStrategy(Strategy):
    """
    Generate definition-style Q&A pairs from entity descriptions.

    Uses entity name + <SEP>-split descriptions from vdb_entities.
    """

    name = "entity_def"

    TEMPLATES = [
        "Qu'est-ce que {entity} dans le cadre du PCG ?",
        "Peux-tu definir {entity} selon le Plan Comptable General ?",
        "Quelle est la definition de {entity} en comptabilite francaise ?",
        "Explique le concept de {entity} dans le PCG 2026.",
    ]

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []
        for name, entity in loader.entities.items():
            descriptions = loader.get_entity_descriptions(name)
            if not descriptions:
                continue

            # Use the longest description as the best answer
            best_desc = max(descriptions, key=len)
            if len(best_desc) < 20:
                continue

            template = random.choice(self.TEMPLATES)
            question = template.format(entity=name)

            pairs.append(FTPair(
                question=question,
                answer=best_desc,
                strategy_name=self.name,
                source_entities=[name],
                metadata={"entity_type": entity.entity_type, "desc_count": len(descriptions)},
            ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Strategy 2: Relational
# ---------------------------------------------------------------------------

class RelationalStrategy(Strategy):
    """
    Generate Q&A pairs about relationships between entities.

    Uses source + relation + target from graph edges.
    """

    name = "relational"

    TEMPLATES = [
        "Quel est le lien entre {src} et {tgt} dans le PCG ?",
        "Comment {src} est-il lie a {tgt} en comptabilite ?",
        "Quelle relation existe entre {src} et {tgt} selon le PCG ?",
    ]

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []
        for rel in loader.relationships:
            desc = rel.description
            if not desc or len(desc) < 20:
                # Fallback to graph edge description
                desc = loader.get_edge_description(rel.src, rel.tgt)
            if not desc or len(desc) < 20:
                continue

            template = random.choice(self.TEMPLATES)
            question = template.format(src=rel.src, tgt=rel.tgt)

            answer = desc
            if rel.keywords:
                answer = f"{desc} (Mots-cles : {', '.join(rel.keywords)})"

            pairs.append(FTPair(
                question=question,
                answer=answer,
                strategy_name=self.name,
                source_entities=[rel.src, rel.tgt],
                metadata={"keywords": rel.keywords},
            ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Strategy 3: Hierarchical
# ---------------------------------------------------------------------------

class HierarchicalStrategy(Strategy):
    """
    Generate Q&A pairs from hierarchical context in text chunks.

    Exploits the [Livre > Titre > Chapitre] context brackets in chunk content.
    """

    name = "hierarchical"

    CONTEXT_RE = re.compile(
        r"\[(?:Livre|LIVRE)\s*:\s*(.+?)(?:\s*>\s*(?:Titre|TITRE)\s*:\s*(.+?))?(?:\s*>\s*(?:Chapitre|CHAPITRE)\s*:\s*(.+?))?\]",
        re.IGNORECASE,
    )

    TEMPLATES = [
        "Que contient la section {section} du PCG ?",
        "Quel est le contenu de {section} dans le Plan Comptable General ?",
        "Resume la section {section} du PCG 2026.",
    ]

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []
        seen_sections = set()

        for chunk_id, chunk in loader.chunks.items():
            match = self.CONTEXT_RE.search(chunk.content)
            if not match:
                continue

            livre = (match.group(1) or "").strip()
            titre = (match.group(2) or "").strip()
            chapitre = (match.group(3) or "").strip()

            # Build section path
            parts = [p for p in [livre, titre, chapitre] if p]
            if not parts:
                continue

            section = " > ".join(parts)
            if section in seen_sections:
                continue
            seen_sections.add(section)

            # Extract content after the context bracket
            bracket_end = match.end()
            content = chunk.content[bracket_end:].strip()
            if len(content) < 50:
                continue

            # Truncate to reasonable length
            if len(content) > 1500:
                content = content[:1500].rsplit(".", 1)[0] + "."

            template = random.choice(self.TEMPLATES)
            question = template.format(section=section)

            pairs.append(FTPair(
                question=question,
                answer=content,
                strategy_name=self.name,
                source_entities=parts,
                metadata={"livre": livre, "titre": titre, "chapitre": chapitre, "chunk_id": chunk_id},
            ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Strategy 4: Comparative
# ---------------------------------------------------------------------------

class ComparativeStrategy(Strategy):
    """
    Generate comparison Q&A pairs between entities of the same type.

    Picks two entities of the same type and contrasts their descriptions.
    """

    name = "comparative"

    TEMPLATES = [
        "Quelle est la difference entre {a} et {b} dans le PCG ?",
        "Compare {a} et {b} selon le Plan Comptable General.",
        "En quoi {a} differe-t-il de {b} en comptabilite francaise ?",
    ]

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []

        # Group entities by type
        by_type: Dict[str, list] = defaultdict(list)
        for name, entity in loader.entities.items():
            descriptions = loader.get_entity_descriptions(name)
            if descriptions and len(max(descriptions, key=len)) >= 30:
                by_type[entity.entity_type].append(entity)

        for entity_type, entities in by_type.items():
            if len(entities) < 2:
                continue

            # Generate pairs for combinations (limit to avoid explosion)
            entity_list = entities[:30]  # cap per type
            for i in range(len(entity_list)):
                for j in range(i + 1, min(i + 4, len(entity_list))):
                    a = entity_list[i]
                    b = entity_list[j]

                    desc_a = max(loader.get_entity_descriptions(a.name), key=len)
                    desc_b = max(loader.get_entity_descriptions(b.name), key=len)

                    template = random.choice(self.TEMPLATES)
                    question = template.format(a=a.name, b=b.name)

                    answer = (
                        f"{a.name} : {desc_a}\n\n"
                        f"{b.name} : {desc_b}"
                    )

                    pairs.append(FTPair(
                        question=question,
                        answer=answer,
                        strategy_name=self.name,
                        source_entities=[a.name, b.name],
                        metadata={"entity_type": entity_type},
                    ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Strategy 5: Multi-hop
# ---------------------------------------------------------------------------

class MultiHopStrategy(Strategy):
    """
    Generate multi-hop reasoning Q&A pairs from graph paths.

    Finds 2-3 edge paths in the graph and constructs questions
    that require traversing the path.
    """

    name = "multihop"

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []
        graph = loader.graph
        nodes = list(graph.nodes())

        if len(nodes) < 3:
            return pairs

        seen_paths = set()
        attempts = 0
        max_attempts = len(nodes) * 10

        while len(pairs) < 200 and attempts < max_attempts:
            attempts += 1

            # Pick a random start node
            start = random.choice(nodes)
            neighbors = list(graph.neighbors(start))
            if not neighbors:
                continue

            # Walk 2-3 hops
            path = [start]
            current = start
            for _ in range(random.randint(1, 2)):
                next_nodes = [n for n in graph.neighbors(current) if n not in path]
                if not next_nodes:
                    break
                current = random.choice(next_nodes)
                path.append(current)

            if len(path) < 3:
                continue

            path_key = tuple(sorted(path))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            # Build chain description
            chain_parts = []
            for k in range(len(path) - 1):
                desc = loader.get_edge_description(path[k], path[k + 1])
                if desc:
                    chain_parts.append(desc)

            if not chain_parts:
                continue

            source = path[0]
            target = path[-1]
            intermediates = path[1:-1]

            question = (
                f"Quel est le lien entre {source} et {target} "
                f"en passant par {', '.join(intermediates)} dans le PCG ?"
            )

            answer = " Ensuite, ".join(chain_parts)

            pairs.append(FTPair(
                question=question,
                answer=answer,
                strategy_name=self.name,
                source_entities=path,
                metadata={"path_length": len(path), "hops": len(path) - 1},
            ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Strategy 6: Chunk QA
# ---------------------------------------------------------------------------

class ChunkQAStrategy(Strategy):
    """
    Generate Q&A pairs by heuristic extraction from chunk text.

    Detects definitions, rules, lists, and article references in chunk content.
    """

    name = "chunk_qa"

    # Patterns for extractable content
    DEFINITION_RE = re.compile(
        r"(?:(?:on entend par|est defini[e]? comme|designe|constitue|s'entend de|correspond a)\s+(.+?)(?:\.|;|$))",
        re.IGNORECASE | re.MULTILINE,
    )

    ARTICLE_RE = re.compile(
        r"(?:Art(?:icle)?\.?\s*(\d+[\-\d]*))[\s\-:]+(.+?)(?:\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    RULE_RE = re.compile(
        r"(?:(?:doivent|doit|sont tenus? de|il est interdit de|ne peuvent? pas|est obligatoire)\s+(.+?)(?:\.|;|$))",
        re.IGNORECASE | re.MULTILINE,
    )

    LIST_RE = re.compile(
        r"(?:comprennent?|incluent?|sont\s*:)\s*\n((?:\s*[-\u2013\u2022]\s*.+\n?)+)",
        re.IGNORECASE,
    )

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []

        for chunk_id, chunk in loader.chunks.items():
            text = chunk.content
            if len(text) < 50:
                continue

            # Extract context section for question context
            context = self._extract_context(text)

            # Strategy 6a: Definitions
            for match in self.DEFINITION_RE.finditer(text):
                definition = match.group(0).strip()
                if len(definition) < 30:
                    continue

                # Find the subject being defined (text before the definition pattern)
                start = max(0, match.start() - 100)
                preceding = text[start:match.start()].strip()
                subject = preceding.rsplit("\n", 1)[-1].strip()
                if not subject or len(subject) < 3:
                    subject = context or "ce concept"

                question = f"Comment est defini(e) {subject} dans le PCG ?"
                pairs.append(FTPair(
                    question=question,
                    answer=definition,
                    strategy_name=self.name,
                    source_entities=[subject],
                    metadata={"sub_strategy": "definition", "chunk_id": chunk_id},
                ))

            # Strategy 6b: Article content
            for match in self.ARTICLE_RE.finditer(text):
                article_num = match.group(1).strip()
                article_content = match.group(2).strip()
                if len(article_content) < 50:
                    continue

                question = f"Que dit l'article {article_num} du PCG ?"
                pairs.append(FTPair(
                    question=question,
                    answer=article_content,
                    strategy_name=self.name,
                    source_entities=[f"Article {article_num}"],
                    metadata={"sub_strategy": "article", "article": article_num, "chunk_id": chunk_id},
                ))

            # Strategy 6c: Rules and obligations
            for match in self.RULE_RE.finditer(text):
                rule = match.group(0).strip()
                if len(rule) < 30:
                    continue

                question = f"Quelle obligation est prevue par le PCG concernant {context or 'cette disposition'} ?"
                pairs.append(FTPair(
                    question=question,
                    answer=rule,
                    strategy_name=self.name,
                    source_entities=[],
                    metadata={"sub_strategy": "rule", "chunk_id": chunk_id},
                ))

            # Strategy 6d: Lists
            for match in self.LIST_RE.finditer(text):
                list_content = match.group(0).strip()
                if len(list_content) < 50:
                    continue

                # Find what the list is about from preceding text
                start = max(0, match.start() - 150)
                preceding = text[start:match.start()].strip()
                subject = preceding.rsplit(".", 1)[-1].strip() or context or "ces elements"

                question = f"Quels sont les elements qui {subject} selon le PCG ?"
                pairs.append(FTPair(
                    question=question,
                    answer=list_content,
                    strategy_name=self.name,
                    source_entities=[],
                    metadata={"sub_strategy": "list", "chunk_id": chunk_id},
                ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs

    @staticmethod
    def _extract_context(text: str) -> str:
        """Extract section context from bracket notation."""
        match = re.search(
            r"\[(?:Livre|LIVRE)\s*:.*?\]",
            text,
            re.IGNORECASE,
        )
        if match:
            ctx = match.group(0)
            # Take the last part of the hierarchy
            parts = ctx.split(">")
            return parts[-1].strip().strip("[]").split(":")[-1].strip()
        return ""


# ---------------------------------------------------------------------------
# Strategy 7: Thematic
# ---------------------------------------------------------------------------

class ThematicStrategy(Strategy):
    """
    Generate thematic overview Q&A pairs from entity type clusters.

    Groups entities by type and generates summary-style questions.
    """

    name = "thematic"

    TYPE_LABELS = {
        "concept": "concepts comptables",
        "content": "sections et contenus",
        "data": "donnees et references",
        "method": "methodes comptables",
        "artifact": "documents et artefacts",
        "organization": "organisations",
        "event": "evenements comptables",
        "location": "localisations",
    }

    def generate(self, loader: KGLoader) -> List[FTPair]:
        pairs = []

        # Group entities by type
        by_type: Dict[str, list] = defaultdict(list)
        for name, entity in loader.entities.items():
            if entity.entity_type and entity.entity_type != "UNKNOWN":
                by_type[entity.entity_type].append(entity)

        for entity_type, entities in by_type.items():
            if len(entities) < 2:
                continue

            label = self.TYPE_LABELS.get(entity_type, entity_type)

            # Overview question
            entity_names = [e.name for e in entities[:20]]
            descriptions = []
            for e in entities[:10]:
                descs = loader.get_entity_descriptions(e.name)
                if descs:
                    descriptions.append(f"- {e.name} : {descs[0]}")

            if not descriptions:
                continue

            question = f"Quels sont les principaux {label} dans le PCG 2026 ?"
            answer = (
                f"Les principaux {label} du PCG 2026 incluent :\n\n"
                + "\n".join(descriptions)
            )

            pairs.append(FTPair(
                question=question,
                answer=answer,
                strategy_name=self.name,
                source_entities=entity_names,
                metadata={"entity_type": entity_type, "count": len(entities)},
            ))

            # Grouped sub-questions by connectivity
            for entity in entities[:15]:
                neighbors = loader.get_neighbors(entity.name)
                if len(neighbors) < 2:
                    continue

                neighbor_descs = []
                for n in neighbors[:8]:
                    desc = loader.get_edge_description(entity.name, n)
                    if desc:
                        neighbor_descs.append(f"- {n} : {desc}")

                if not neighbor_descs:
                    continue

                question = f"Quels elements sont lies a {entity.name} dans le PCG ?"
                answer = (
                    f"Les elements lies a {entity.name} dans le PCG sont :\n\n"
                    + "\n".join(neighbor_descs)
                )

                pairs.append(FTPair(
                    question=question,
                    answer=answer,
                    strategy_name=self.name,
                    source_entities=[entity.name] + neighbors[:8],
                    metadata={"entity_type": entity_type, "neighbors": len(neighbors)},
                ))

        logger.info(f"[{self.name}] Generated {len(pairs)} pairs")
        return pairs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, type] = {
    "entity_def": EntityDefinitionStrategy,
    "relational": RelationalStrategy,
    "hierarchical": HierarchicalStrategy,
    "comparative": ComparativeStrategy,
    "multihop": MultiHopStrategy,
    "chunk_qa": ChunkQAStrategy,
    "thematic": ThematicStrategy,
}


def get_strategies(names: Optional[List[str]] = None) -> List[Strategy]:
    """
    Get strategy instances by name.

    Args:
        names: List of strategy names. None = all strategies.

    Returns:
        List of Strategy instances.
    """
    if names is None:
        names = list(STRATEGIES.keys())

    instances = []
    for name in names:
        cls = STRATEGIES.get(name)
        if cls is None:
            logger.warning(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
            continue
        instances.append(cls())
    return instances
