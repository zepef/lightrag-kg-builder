"""
Code du Travail Chunking Profile

Defines regex patterns and hierarchy for French labor law documents.
Structure: Partie > Livre > Titre > Chapitre > Section > Article
"""

from ..base import ChunkingProfile

CODE_TRAVAIL_PROFILE = ChunkingProfile(
    name="code-travail",
    patterns={
        'partie': r'^(?:Premi[eè]re|Deuxi[eè]me|Troisi[eè]me|Quatri[eè]me|Cinqui[eè]me|Sixi[eè]me|Septi[eè]me|Huiti[eè]me)\s+partie\s*[:\-\u2013]?\s*(.+)$',
        'livre': r'^Livre\s+([IVX]+(?:er)?|\d+(?:er)?)\s*[:\-\u2013]?\s*(.+)$',
        'titre': r'^Titre\s+([IVX]+(?:er)?|\d+(?:er)?)\s*[:\-\u2013]?\s*(.+)$',
        'chapitre': r'^Chapitre\s+([IVX]+(?:er)?|\d+(?:er)?|pr[eé]liminaire|unique)\s*[:\-\u2013.]?\s*(.*)$',
        'section': r'^Section\s+(\d+)\s*[:\-\u2013]?\s*(.+)$',
        'sous_section': r'^Sous-section\s+(\d+)\s*[:\-\u2013]?\s*(.+)$',
        'article': r'^Article\s+[LRD]\.?\s*(\d+[\-\d]*(?:-\d+)*)\s*(.*)$',
    },
    hierarchy=['partie', 'livre', 'titre', 'chapitre', 'section', 'sous_section'],
    atomic_units=['article'],
    context_format="[{partie} > {livre} > {titre} > {chapitre}]",
    split_on=['livre', 'titre', 'chapitre'],
)
