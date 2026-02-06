"""
Chunking Profile System

Defines the ChunkingProfile dataclass and ChunkResult for domain-specific
document chunking. Profiles encapsulate regex patterns and hierarchy rules
that vary between legal document types (PCG, Code Civil, Code des Impots, etc.).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChunkingProfile:
    """
    Configuration for domain-specific document chunking.

    A profile defines the structural patterns (regex), hierarchy levels,
    atomic units (elements that should never be split), and context formatting
    for a specific document type.

    Example:
        PCG_PROFILE = ChunkingProfile(
            name="pcg",
            patterns={
                'livre': r'^Livre\\s+([IVX]+|\\d+)\\s*[:\\-\\u2013]?\\s*(.+)$',
                'titre': r'^TITRE\\s+([IVX]+)\\s*[\\u2013\\-:]\\s*(.+)$',
                ...
            },
            hierarchy=['livre', 'titre', 'chapitre', 'section', 'sous_section'],
            atomic_units=['article', 'account_def'],
            context_format="[{livre} > {titre} > {chapitre}]"
        )
    """
    name: str
    patterns: Dict[str, str]           # level_name -> regex pattern string
    hierarchy: List[str]               # ordered hierarchy levels (top to bottom)
    atomic_units: List[str]            # keep these together (articles, accounts, etc.)
    context_format: str                # e.g. "[{livre} > {titre} > {chapitre}]"
    split_on: Optional[List[str]] = None  # levels that force a new chunk (defaults to hierarchy[1:3])

    def __post_init__(self):
        if self.split_on is None:
            # Default: split on second and third hierarchy levels
            self.split_on = self.hierarchy[1:4] if len(self.hierarchy) > 1 else self.hierarchy

    def get_compiled_patterns(self) -> Dict[str, re.Pattern]:
        """Return compiled regex patterns."""
        compiled = {}
        for name, pattern_str in self.patterns.items():
            flags = re.MULTILINE
            # Auto-add IGNORECASE for livre-level patterns
            if name == self.hierarchy[0] if self.hierarchy else False:
                flags |= re.IGNORECASE
            compiled[name] = re.compile(pattern_str, flags)
        return compiled


@dataclass
class ChunkResult:
    """Result from chunking a single document section."""
    chunk_id: str
    chunk_type: str     # e.g. 'article', 'section', 'account_group', 'mixed'
    content: str
    context: Dict[str, Optional[str]]  # hierarchy level -> value
    atomic_ids: List[str] = field(default_factory=list)  # article IDs, account codes, etc.
    char_count: int = 0

    def get_context_prefix(self, profile: ChunkingProfile) -> str:
        """Generate context prefix string from hierarchy context."""
        parts = []
        for level in profile.hierarchy:
            val = self.context.get(level)
            if val:
                # Capitalize level name for display
                display = level.replace('_', '-').capitalize()
                parts.append(f"{display}: {val}")
        return " > ".join(parts) if parts else ""

    def to_text(self, profile: ChunkingProfile, include_context: bool = True) -> str:
        """Convert chunk to text with optional context prefix."""
        if include_context:
            prefix = self.get_context_prefix(profile)
            if prefix:
                return f"[{prefix}]\n\n{self.content}"
        return self.content

    def to_dict(self, profile: ChunkingProfile, include_context: bool = True) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.chunk_id,
            'text': self.to_text(profile, include_context=include_context),
            'metadata': {
                'type': self.chunk_type,
                **{level: self.context.get(level) for level in profile.hierarchy},
                'atomic_ids': self.atomic_ids,
                'char_count': self.char_count,
            }
        }
