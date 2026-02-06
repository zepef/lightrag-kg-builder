"""
PCG (Plan Comptable General) Chunking Profile

Defines regex patterns and hierarchy for French accounting standard documents.
Structure: Livre > TITRE > CHAPITRE > Section > Sous-section > Article > Account
"""

from ..base import ChunkingProfile

PCG_PROFILE = ChunkingProfile(
    name="pcg",
    patterns={
        'livre': r'^Livre\s+([IVX]+|\d+)\s*[:\-\u2013]?\s*(.+)$',
        'titre': r'^TITRE\s+([IVX]+)\s*[\u2013\-:]\s*(.+)$',
        'chapitre': r'^CHAPITRE\s+([IVX]+)\s*[\u2013\-:]\s*(.+)$',
        'section': r'^Section\s+(\d+)\s*[\u2013\-:]\s*(.+)$',
        'sous_section': r'^Sous-section\s+(\d+)\s*[\u2013\-:]\s*(.+)$',
        'article': r'^Article\s+(\d{3}-\d+(?:-\d+)?)\s*[:\-\u2013]?\s*(.*)$',
        'account_def': r'^(\d{3,5})\s+([A-Z\u00c0\u00c2\u00c4\u00c9\u00c8\u00ca\u00cb\u00cf\u00ce\u00d4\u00d9\u00db\u00dc\u00c7].+)$',
    },
    hierarchy=['livre', 'titre', 'chapitre', 'section', 'sous_section'],
    atomic_units=['article', 'account_def'],
    context_format="[{livre} > {titre} > {chapitre}]",
    split_on=['titre', 'chapitre', 'section'],
)
