"""
PDF Extractor
Extract text from PDF files preserving structure markers.

Uses PyMuPDF (fitz) for reliable extraction. Generic — works with any PDF,
with optional cleanup patterns passed via configuration.
"""

import re
from pathlib import Path
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF required: pip install PyMuPDF")


@dataclass
class ExtractionResult:
    """Result of PDF extraction"""
    text: str
    pages: int
    chars: int
    source_files: List[str]


class PdfExtractor:
    """
    Extract text from PDF files with optional cleanup.

    Features:
    - Preserves section headers and structure markers
    - Configurable cleanup patterns for domain-specific artifacts
    - Handles multi-file extraction
    - Progress callback for UI feedback
    """

    # Default cleanup patterns (generic PDF artifacts)
    DEFAULT_CLEANUP_PATTERNS: List[Tuple[str, str]] = [
        (r'\n\s*\d+\s*\n', '\n'),       # Page numbers
        (r'\n{3,}', '\n\n'),              # Multiple newlines
        (r'[ \t]+\n', '\n'),              # Trailing spaces
    ]

    def __init__(self, cleanup_patterns: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize extractor.

        Args:
            cleanup_patterns: List of (regex_pattern, replacement) tuples.
                              If None, uses DEFAULT_CLEANUP_PATTERNS.
                              Pass an empty list to disable cleanup.
        """
        self.cleanup_patterns = cleanup_patterns if cleanup_patterns is not None else self.DEFAULT_CLEANUP_PATTERNS
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        Set progress callback for UI updates.

        Args:
            callback: Function(current, total, message)
        """
        self.progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def extract_single(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted and cleaned text
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        text_parts = []

        self._report_progress(0, total_pages, f"Extracting {pdf_path.name}")

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            text_parts.append(text)

            if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                self._report_progress(
                    page_num + 1,
                    total_pages,
                    f"{pdf_path.name}: page {page_num + 1}/{total_pages}"
                )

        doc.close()

        raw_text = '\n'.join(text_parts)
        cleaned = self._clean_text(raw_text)

        return cleaned

    def extract_all(self, sources_dir: Path) -> ExtractionResult:
        """
        Extract text from all PDFs in a directory.

        Args:
            sources_dir: Directory containing PDF files

        Returns:
            ExtractionResult with combined text and metadata
        """
        sources_dir = Path(sources_dir)
        if not sources_dir.exists():
            raise FileNotFoundError(f"Sources directory not found: {sources_dir}")

        pdf_files = sorted(sources_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {sources_dir}")

        all_texts = []
        total_pages = 0
        source_files = []

        for pdf_path in pdf_files:
            text = self.extract_single(pdf_path)
            all_texts.append(f"\n\n{'='*60}\n# SOURCE: {pdf_path.name}\n{'='*60}\n\n{text}")
            source_files.append(pdf_path.name)

            doc = fitz.open(pdf_path)
            total_pages += len(doc)
            doc.close()

        combined = '\n\n'.join(all_texts)
        combined = self._final_cleanup(combined)

        return ExtractionResult(
            text=combined,
            pages=total_pages,
            chars=len(combined),
            source_files=source_files
        )

    def _clean_text(self, text: str) -> str:
        """Apply cleanup patterns to extracted text"""
        result = text
        for pattern, replacement in self.cleanup_patterns:
            result = re.sub(pattern, replacement, result)
        return result.strip()

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass for combined text"""
        # Normalize unicode
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')
        text = text.replace('\u00a0', ' ')

        # Remove null bytes and problematic chars
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
