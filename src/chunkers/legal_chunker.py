"""
Legal Document Chunker

Generic chunker for structured legal documents (accounting standards, civil code,
tax code, etc.). Uses ChunkingProfile to adapt to different document types.

Features:
- Respects hierarchical boundaries defined by the profile
- Keeps atomic units together (articles, accounts, etc.)
- Adds context prefix to each chunk
- Configurable max/min chunk size with smart splitting
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from .base import ChunkingProfile, ChunkResult


class LegalDocumentChunker:
    """
    Semantic chunker for structured legal documents.

    Takes a ChunkingProfile that defines the document structure patterns,
    then splits text into semantically meaningful chunks that respect
    the document's hierarchy.
    """

    def __init__(
        self,
        profile: ChunkingProfile,
        max_chunk_size: int = 4000,
        min_chunk_size: int = 500,
        overlap_size: int = 200,
        include_context: bool = True,
    ):
        """
        Initialize chunker with a profile and configuration.

        Args:
            profile: ChunkingProfile defining patterns and hierarchy
            max_chunk_size: Maximum characters per chunk (default 4000 ~1000 tokens)
            min_chunk_size: Minimum characters to form a chunk (default 500)
            overlap_size: Characters to overlap between chunks when splitting (default 200)
            include_context: Whether to add hierarchical context prefix (default True)
        """
        self.profile = profile
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.include_context = include_context
        self._compiled_patterns = profile.get_compiled_patterns()

    def parse_structure(self, text: str) -> List[dict]:
        """Parse document into structural elements with positions."""
        elements = []

        for elem_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                elements.append({
                    'type': elem_type,
                    'start': match.start(),
                    'end': match.end(),
                    'id': match.group(1),
                    'title': match.group(2).strip() if match.lastindex >= 2 else '',
                    'full_match': match.group(0)
                })

        elements.sort(key=lambda x: x['start'])
        return elements

    def extract_sections(self, text: str) -> List[Tuple[dict, dict, str]]:
        """Extract text sections with their structural context."""
        elements = self.parse_structure(text)
        sections = []

        # Initialize context with all hierarchy levels as None
        context = {level: None for level in self.profile.hierarchy}

        for i, elem in enumerate(elements):
            elem_type = elem['type']

            # Update context based on element type
            if elem_type in self.profile.hierarchy:
                idx = self.profile.hierarchy.index(elem_type)
                context[elem_type] = f"{elem['id']} - {elem['title']}"
                # Clear all levels below this one
                for lower_level in self.profile.hierarchy[idx + 1:]:
                    context[lower_level] = None

            # Extract content until next element
            start = elem['start']
            end = elements[i + 1]['start'] if i + 1 < len(elements) else len(text)
            content = text[start:end].strip()

            if content:
                sections.append((context.copy(), elem, content))

        return sections

    def create_chunks(self, text: str) -> List[ChunkResult]:
        """
        Create semantic chunks from document text.

        Strategy:
        1. Parse structural elements using profile patterns
        2. Group atomic units together (articles, accounts, etc.)
        3. Split at hierarchy boundaries defined by profile.split_on
        4. Split large sections at paragraph boundaries
        5. Add context prefix to each chunk
        """
        chunks = []
        chunk_counter = 0

        sections = self.extract_sections(text)
        current_content = []
        current_context = None
        current_atomic_ids = []

        for context, elem, content in sections:
            should_split = False

            # New major section starts new chunk
            if elem['type'] in self.profile.split_on and current_content:
                should_split = True

            # Size limit reached
            total_size = sum(len(c) for c in current_content) + len(content)
            if total_size > self.max_chunk_size and current_content:
                should_split = True

            if should_split:
                chunk = self._create_chunk(
                    chunk_counter,
                    current_content,
                    current_context,
                    current_atomic_ids,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1

                current_content = []
                current_atomic_ids = []

            current_content.append(content)
            current_context = context

            # Track atomic unit IDs
            if elem['type'] in self.profile.atomic_units:
                current_atomic_ids.append(elem['id'])

            # Also find atomic units within content
            for atomic_type in self.profile.atomic_units:
                if atomic_type in self._compiled_patterns:
                    for match in self._compiled_patterns[atomic_type].finditer(content):
                        aid = match.group(1)
                        if aid not in current_atomic_ids:
                            current_atomic_ids.append(aid)

        # Last chunk
        if current_content:
            chunk = self._create_chunk(
                chunk_counter,
                current_content,
                current_context,
                current_atomic_ids,
            )
            if chunk:
                chunks.append(chunk)

        # Post-process: split chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if chunk.char_count > self.max_chunk_size * 1.5:
                final_chunks.extend(self._split_large_chunk(chunk, len(final_chunks)))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def chunk_document(self, text: str) -> List[dict]:
        """
        Convenience method: chunk text and return list of dicts.

        Returns:
            List of chunk dictionaries with 'id', 'text', 'metadata' keys
        """
        chunks = self.create_chunks(text)
        return [chunk.to_dict(self.profile, include_context=self.include_context) for chunk in chunks]

    def _create_chunk(
        self,
        idx: int,
        content_parts: List[str],
        context: Optional[dict],
        atomic_ids: List[str],
    ) -> Optional[ChunkResult]:
        """Create a ChunkResult from collected content."""
        content = "\n\n".join(content_parts)
        if len(content) < self.min_chunk_size:
            return None

        # Determine chunk type from atomic IDs
        if atomic_ids:
            # Check if the IDs look like article references or account codes
            has_articles = any('-' in aid for aid in atomic_ids)
            has_accounts = any(aid.isdigit() for aid in atomic_ids)
            if has_articles:
                chunk_type = 'article'
            elif has_accounts:
                chunk_type = 'account_group'
            else:
                chunk_type = 'mixed'
        else:
            chunk_type = 'section'

        return ChunkResult(
            chunk_id=f"{self.profile.name}-{idx:04d}",
            chunk_type=chunk_type,
            content=content,
            context=context or {level: None for level in self.profile.hierarchy},
            atomic_ids=atomic_ids,
            char_count=len(content),
        )

    def _split_large_chunk(self, chunk: ChunkResult, start_idx: int) -> List[ChunkResult]:
        """Split a large chunk at paragraph boundaries."""
        paragraphs = chunk.content.split('\n\n')
        sub_chunks = []
        current_parts = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.max_chunk_size and current_parts:
                sub_content = '\n\n'.join(current_parts)
                sub_chunk = ChunkResult(
                    chunk_id=f"{self.profile.name}-{start_idx + len(sub_chunks):04d}",
                    chunk_type=chunk.chunk_type,
                    content=sub_content,
                    context=chunk.context.copy(),
                    atomic_ids=[],
                    char_count=len(sub_content),
                )
                sub_chunks.append(sub_chunk)

                current_parts = [para]
                current_size = para_size
            else:
                current_parts.append(para)
                current_size += para_size

        if current_parts:
            sub_content = '\n\n'.join(current_parts)
            sub_chunk = ChunkResult(
                chunk_id=f"{self.profile.name}-{start_idx + len(sub_chunks):04d}",
                chunk_type=chunk.chunk_type,
                content=sub_content,
                context=chunk.context.copy(),
                atomic_ids=[],
                char_count=len(sub_content),
            )
            sub_chunks.append(sub_chunk)

        return sub_chunks if sub_chunks else [chunk]
