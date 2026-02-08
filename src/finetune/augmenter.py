"""
Data augmentation through paraphrasing.

Generates N variants of each Q&A pair to artificially expand the training set.
Two modes:
  - Template-based (default): synonym substitution + structural transforms, no LLM needed
  - LLM-powered (--llm): richer rephrasing via a local model (vLLM/Ollama)
"""

import copy
import logging
import random
import re
from typing import Dict, List, Optional, Tuple

from .strategies import FTPair

logger = logging.getLogger(__name__)


# =============================================================================
# French accounting synonym groups for template-based paraphrasing
# =============================================================================

# Question-level synonyms: interchangeable question phrasings
QUESTION_SYNONYMS: List[List[str]] = [
    ["Qu'est-ce que", "Que designe", "Comment definit-on", "Quelle est la notion de"],
    ["Peux-tu definir", "Donne la definition de", "Definis", "Precise ce qu'on entend par"],
    ["Quelle est la definition de", "Comment est defini(e)", "Que signifie", "Quel est le sens de"],
    ["Explique le concept de", "Decris le principe de", "Presente la notion de", "En quoi consiste"],
    ["Quel est le lien entre", "Quelle relation existe entre", "Comment sont lies", "Quel rapport y a-t-il entre"],
    ["Quelle est la difference entre", "En quoi differe", "Compare", "Distingue"],
    ["Que contient", "Quel est le contenu de", "Que prevoit", "De quoi traite"],
    ["Quels sont les principaux", "Enumere les", "Cite les", "Liste les"],
    ["Quels elements sont lies a", "Qu'est-ce qui se rapporte a", "Quels composants concernent"],
    ["Que dit l'article", "Que prevoit l'article", "Quel est le contenu de l'article", "Que stipule l'article"],
    ["Quelle obligation est prevue", "Quelle regle s'applique", "Que doit-on respecter concernant"],
    ["Resume la section", "Synthetise le contenu de", "Donne un apercu de", "Presente brievement"],
]

# Domain-specific synonym pairs (term -> [alternatives])
TERM_SYNONYMS: Dict[str, List[str]] = {
    "comptabilite": ["comptabilite", "tenue des comptes", "science comptable"],
    "comptes annuels": ["comptes annuels", "etats financiers annuels", "documents comptables annuels"],
    "bilan": ["bilan", "etat de la situation financiere", "bilan comptable"],
    "compte de resultat": ["compte de resultat", "etat du resultat", "compte de pertes et profits"],
    "annexe": ["annexe", "notes annexes", "informations complementaires"],
    "actif": ["actif", "elements d'actif", "ressources economiques"],
    "passif": ["passif", "elements de passif", "obligations"],
    "charges": ["charges", "depenses", "couts"],
    "produits": ["produits", "revenus", "recettes"],
    "amortissement": ["amortissement", "depreciation systematique", "repartition du cout"],
    "provision": ["provision", "dotation", "charge provisionnee"],
    "immobilisation": ["immobilisation", "actif immobilise", "bien durable"],
    "creance": ["creance", "droit de creance", "actif a recevoir"],
    "dette": ["dette", "obligation financiere", "engagement"],
    "capitaux propres": ["capitaux propres", "fonds propres", "situation nette"],
    "PCG": ["PCG", "Plan Comptable General", "le referentiel comptable francais"],
    "dans le cadre du PCG": ["dans le cadre du PCG", "selon le PCG", "au sens du PCG", "d'apres le PCG"],
    "en comptabilite francaise": ["en comptabilite francaise", "dans le referentiel francais", "selon les normes francaises"],
    "dans le PCG 2026": ["dans le PCG 2026", "selon le PCG 2026", "au titre du PCG 2026"],
}

# Answer-level connector synonyms for restructuring
CONNECTOR_SWAPS: List[Tuple[str, List[str]]] = [
    ("En outre,", ["De plus,", "Par ailleurs,", "Egalement,", "De surcroit,"]),
    ("De plus,", ["En outre,", "Par ailleurs,", "Aussi,", "Egalement,"]),
    ("Cependant,", ["Toutefois,", "Neanmoins,", "Pourtant,", "En revanche,"]),
    ("Ainsi,", ["De cette maniere,", "Par consequent,", "En consequence,", "Des lors,"]),
    ("C'est-a-dire", ["autrement dit", "en d'autres termes", "soit"]),
    ("Ensuite,", ["Puis,", "Apres quoi,", "Par la suite,"]),
    (" est ", [" correspond a ", " designe ", " represente "]),
    (" comprend ", [" inclut ", " englobe ", " integre "]),
    (" doit ", [" est tenu(e) de ", " a l'obligation de "]),
    (" permet de ", [" rend possible ", " offre la possibilite de "]),
]


# =============================================================================
# Template-based augmenter (no LLM)
# =============================================================================

class TemplateAugmenter:
    """
    Generate paraphrased variants using synonym substitution and structural transforms.

    For each pair, produces up to `n` variants by:
    1. Swapping question phrasing with synonym groups
    2. Substituting domain-specific terms
    3. Swapping answer connectors
    """

    def __init__(self, n: int = 3, seed: Optional[int] = None):
        self.n = n
        self.rng = random.Random(seed)

    def augment(self, pairs: List[FTPair]) -> List[FTPair]:
        """
        Augment pairs with paraphrased variants.

        Returns original pairs + generated variants.
        """
        augmented = []
        total_variants = 0

        for pair in pairs:
            augmented.append(pair)  # keep original
            variants = self._generate_variants(pair)
            augmented.extend(variants)
            total_variants += len(variants)

        logger.info(
            f"Augmentation: {len(pairs)} originals + {total_variants} variants "
            f"= {len(augmented)} total ({self.n}x target)"
        )
        return augmented

    def _generate_variants(self, pair: FTPair) -> List[FTPair]:
        """Generate up to N paraphrased variants of a single pair."""
        variants = []
        seen_questions = {pair.question.lower().strip()}

        for _ in range(self.n * 3):  # try more times than needed to fill N slots
            if len(variants) >= self.n:
                break

            new_q = self._paraphrase_question(pair.question)
            new_a = self._paraphrase_answer(pair.answer)

            # Skip if question is too similar to original or already generated
            q_normalized = new_q.lower().strip()
            if q_normalized in seen_questions:
                continue
            if new_q == pair.question and new_a == pair.answer:
                continue

            seen_questions.add(q_normalized)
            variants.append(FTPair(
                question=new_q,
                answer=new_a,
                strategy_name=pair.strategy_name,
                source_entities=pair.source_entities.copy(),
                metadata={**pair.metadata, "augmented": True, "original_q": pair.question[:80]},
            ))

        return variants

    def _paraphrase_question(self, question: str) -> str:
        """Swap question phrasing using synonym groups."""
        result = question

        # Try each synonym group
        for group in QUESTION_SYNONYMS:
            for phrase in group:
                if phrase in result:
                    alternatives = [p for p in group if p != phrase]
                    if alternatives:
                        result = result.replace(phrase, self.rng.choice(alternatives), 1)
                    break  # only one swap per group

        # Randomly swap a domain term (30% chance per term)
        for term, synonyms in TERM_SYNONYMS.items():
            if term in result and self.rng.random() < 0.3:
                alt = self.rng.choice([s for s in synonyms if s != term])
                result = result.replace(term, alt, 1)

        return result

    def _paraphrase_answer(self, answer: str) -> str:
        """Swap connectors and lightly restructure the answer."""
        result = answer

        # Swap 1-2 connectors randomly
        swaps_done = 0
        shuffled_connectors = self.CONNECTOR_SWAPS_SHUFFLED()
        for original, alternatives in shuffled_connectors:
            if swaps_done >= 2:
                break
            if original in result:
                result = result.replace(original, self.rng.choice(alternatives), 1)
                swaps_done += 1

        # Randomly swap a domain term in the answer (20% chance)
        for term, synonyms in TERM_SYNONYMS.items():
            if term in result and self.rng.random() < 0.2:
                alt = self.rng.choice([s for s in synonyms if s != term])
                result = result.replace(term, alt, 1)

        return result

    def CONNECTOR_SWAPS_SHUFFLED(self) -> List[Tuple[str, List[str]]]:
        """Return connector swaps in random order."""
        swaps = CONNECTOR_SWAPS.copy()
        self.rng.shuffle(swaps)
        return swaps


# =============================================================================
# LLM-powered augmenter (optional)
# =============================================================================

class LLMAugmenter:
    """
    Generate paraphrased variants using a local LLM (vLLM or Ollama).

    Sends each pair to the model with a rephrasing prompt.
    Produces richer, more natural paraphrases than template-based.
    """

    REPHRASE_PROMPT = (
        "Tu es un assistant specialise en comptabilite francaise. "
        "Reformule la question et la reponse ci-dessous en conservant exactement "
        "le meme sens et les memes informations techniques. "
        "Utilise une formulation differente, des synonymes, et une structure de phrase differente. "
        "Reponds UNIQUEMENT avec le JSON suivant, sans texte supplementaire :\n"
        '{{"question": "...", "answer": "..."}}\n\n'
        "Question originale : {question}\n"
        "Reponse originale : {answer}"
    )

    def __init__(
        self,
        n: int = 3,
        api_url: str = "http://localhost:8000/v1",
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.8,
    ):
        self.n = n
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def augment(self, pairs: List[FTPair]) -> List[FTPair]:
        """Augment pairs using LLM rephrasing. Requires running LLM server."""
        import json

        try:
            import httpx
        except ImportError:
            logger.error("httpx required for LLM augmentation: pip install httpx")
            return pairs

        augmented = list(pairs)  # keep originals
        total_variants = 0
        errors = 0

        client = httpx.Client(timeout=60.0)

        for i, pair in enumerate(pairs):
            if i % 50 == 0:
                logger.info(f"LLM augmenting: {i}/{len(pairs)} ({total_variants} variants so far)")

            for attempt in range(self.n):
                try:
                    prompt = self.REPHRASE_PROMPT.format(
                        question=pair.question,
                        answer=pair.answer[:1000],  # cap answer length for prompt
                    )

                    resp = client.post(
                        f"{self.api_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.temperature + (attempt * 0.1),  # vary temp per attempt
                            "max_tokens": 1500,
                        },
                    )
                    resp.raise_for_status()

                    content = resp.json()["choices"][0]["message"]["content"].strip()

                    # Parse JSON response
                    # Handle cases where model wraps in ```json
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                    parsed = json.loads(content)
                    new_q = parsed.get("question", "").strip()
                    new_a = parsed.get("answer", "").strip()

                    if not new_q or not new_a or len(new_a) < 30:
                        continue

                    augmented.append(FTPair(
                        question=new_q,
                        answer=new_a,
                        strategy_name=pair.strategy_name,
                        source_entities=pair.source_entities.copy(),
                        metadata={**pair.metadata, "augmented": True, "augment_method": "llm"},
                    ))
                    total_variants += 1

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"LLM augmentation error: {e}")

        client.close()

        if errors > 5:
            logger.warning(f"LLM augmentation: {errors - 5} additional errors suppressed")

        total_attempts = len(pairs) * self.n
        logger.info(
            f"LLM augmentation: {len(pairs)} originals + {total_variants} variants "
            f"= {len(augmented)} total ({errors}/{total_attempts} attempts failed)"
        )
        return augmented


def get_augmenter(
    n: int,
    use_llm: bool = False,
    llm_url: str = "http://localhost:8000/v1",
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    seed: Optional[int] = None,
):
    """Factory function to get the appropriate augmenter."""
    if use_llm:
        return LLMAugmenter(n=n, api_url=llm_url, model=llm_model)
    return TemplateAugmenter(n=n, seed=seed)
