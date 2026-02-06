"""
Execution Journal
Structured logging with timestamps, phases, and metrics for KG build pipeline.

Features:
- Append-only JSONL format for crash resilience
- Phase tracking with durations
- Resume state detection
- Execution summary generation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import time


@dataclass
class JournalEntry:
    """A single journal entry"""
    timestamp: str
    phase: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None

    @classmethod
    def now(cls, phase: str, action: str, details: Dict[str, Any] = None, duration_ms: int = None):
        """Create entry with current timestamp"""
        return cls(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            action=action,
            details=details or {},
            duration_ms=duration_ms
        )


class ExecutionJournal:
    """
    Persistent execution log for debugging and audit.

    Uses append-only JSONL format to survive crashes.
    Each entry is a complete JSON object on its own line.
    """

    def __init__(self, journal_path: Path):
        """
        Initialize journal.

        Args:
            journal_path: Path to JSONL file (will be created if not exists)
        """
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._phase_starts: Dict[str, float] = {}

    def log(
        self,
        phase: str,
        action: str,
        details: Dict[str, Any] = None,
        duration_ms: int = None
    ):
        """
        Append entry to journal.

        Args:
            phase: Phase name (pdf_extraction, chunking, embedding, llm, graph, pipeline)
            action: Action type (start, progress, complete, error, resume, checkpoint)
            details: Additional details dict
            duration_ms: Optional duration in milliseconds
        """
        entry = JournalEntry.now(phase, action, details, duration_ms)

        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

    def start_phase(self, phase: str, details: Dict[str, Any] = None):
        """Start a phase and track its start time"""
        self._phase_starts[phase] = time.time()
        self.log(phase, 'start', details)

    def complete_phase(self, phase: str, details: Dict[str, Any] = None):
        """Complete a phase with automatic duration calculation"""
        duration_ms = None
        if phase in self._phase_starts:
            duration_ms = int((time.time() - self._phase_starts[phase]) * 1000)
            del self._phase_starts[phase]

        self.log(phase, 'complete', details, duration_ms)

    def error_phase(self, phase: str, error: str, details: Dict[str, Any] = None):
        """Log an error for a phase"""
        err_details = details or {}
        err_details['error'] = error
        self.log(phase, 'error', err_details)

    def progress(self, phase: str, current: int, total: int, **extra):
        """Log progress update"""
        details = {
            'current': current,
            'total': total,
            'pct': round(current / total * 100, 1) if total > 0 else 0,
            **extra
        }
        self.log(phase, 'progress', details)

    def checkpoint(self, details: Dict[str, Any]):
        """Log a checkpoint for resume"""
        self.log('pipeline', 'checkpoint', details)

    def get_entries(self) -> List[JournalEntry]:
        """Read all journal entries"""
        if not self.path.exists():
            return []

        entries = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(JournalEntry(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue

        return entries

    def get_last_state(self) -> Dict[str, Any]:
        """
        Get last known state for resume.

        Returns:
            Dict with phase completion status and last checkpoint/error info
        """
        entries = self.get_entries()

        state = {
            'pdf_complete': False,
            'chunking_complete': False,
            'embedding_complete': False,
            'llm_complete': False,
            'graph_complete': False,
            'pipeline_complete': False,
            'last_checkpoint': None,
            'last_error': None,
            'sections_completed': 0,
        }

        for entry in entries:
            if entry.action == 'complete':
                if entry.phase == 'pdf_extraction':
                    state['pdf_complete'] = True
                elif entry.phase == 'chunking':
                    state['chunking_complete'] = True
                elif entry.phase == 'embedding':
                    state['embedding_complete'] = True
                elif entry.phase == 'llm':
                    state['llm_complete'] = True
                elif entry.phase == 'graph':
                    state['graph_complete'] = True
                elif entry.phase == 'pipeline':
                    state['pipeline_complete'] = True

            elif entry.action == 'checkpoint':
                state['last_checkpoint'] = entry.details

            elif entry.action == 'error':
                state['last_error'] = {
                    'phase': entry.phase,
                    'timestamp': entry.timestamp,
                    'details': entry.details
                }

            elif entry.action == 'progress' and entry.phase == 'section':
                if 'current' in entry.details:
                    state['sections_completed'] = entry.details['current']

        return state

    def summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        entries = self.get_entries()

        if not entries:
            return {'status': 'no_data', 'entries': 0}

        phases = {}
        errors = []

        for entry in entries:
            if entry.phase not in phases:
                phases[entry.phase] = {
                    'started': False,
                    'completed': False,
                    'duration_ms': None,
                    'details': {}
                }

            if entry.action == 'start':
                phases[entry.phase]['started'] = True
                phases[entry.phase]['start_time'] = entry.timestamp

            elif entry.action == 'complete':
                phases[entry.phase]['completed'] = True
                phases[entry.phase]['duration_ms'] = entry.duration_ms
                phases[entry.phase]['details'] = entry.details

            elif entry.action == 'error':
                errors.append({
                    'phase': entry.phase,
                    'timestamp': entry.timestamp,
                    'error': entry.details.get('error', 'Unknown')
                })

        first_entry = entries[0] if entries else None
        last_entry = entries[-1] if entries else None

        return {
            'status': 'complete' if phases.get('pipeline', {}).get('completed') else 'incomplete',
            'entries': len(entries),
            'phases': phases,
            'errors': errors,
            'error_count': len(errors),
            'start_time': first_entry.timestamp if first_entry else None,
            'end_time': last_entry.timestamp if last_entry else None,
        }

    def clear(self):
        """Clear journal (for fresh start)"""
        if self.path.exists():
            self.path.unlink()
