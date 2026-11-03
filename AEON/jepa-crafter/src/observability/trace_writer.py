"""Append-only JSONL trace writer with checksumming and crash handling."""

import hashlib
import json
import os
from pathlib import Path
from typing import Generator, List, Optional

from .events import Event


class TraceWriter:
    """Thread-safe append-only trace writer for structured events."""

    def __init__(self, trace_path: str, enabled: bool = True):
        self.trace_path = Path(trace_path)
        self.enabled = enabled
        self._file = None
        self._event_count = 0
        self._hasher = hashlib.sha256()

        if self.enabled:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.trace_path, "a", encoding="utf-8")

    def write_event(self, event: Event) -> None:
        if not self.enabled or self._file is None:
            return

        line = event.to_json() + "\n"
        self._file.write(line)
        self._file.flush()
        self._hasher.update(line.encode("utf-8"))
        self._event_count += 1

    def flush(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> str:
        """Close the trace file and write the companion SHA-256 checksum file."""
        if self._file and not self._file.closed:
            self.flush()
            self._file.close()

        final_checksum = self._compute_file_checksum(self.trace_path)
        checksum_path = self.trace_path.with_suffix(".sha256")
        if self.trace_path.exists():
            checksum_path.write_text(final_checksum, encoding="utf-8")
        return final_checksum

    @staticmethod
    def _compute_file_checksum(path: Path) -> str:
        if not path.exists():
            return ""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def verify_checksum(cls, trace_path: str) -> bool:
        p = Path(trace_path)
        chk_p = p.with_suffix(".sha256")
        if not p.exists() or not chk_p.exists():
            return False
        expected = chk_p.read_text(encoding="utf-8").strip()
        actual = cls._compute_file_checksum(p)
        return expected == actual

    @classmethod
    def read_events(cls, trace_path: str) -> Generator[dict, None, None]:
        p = Path(trace_path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @property
    def event_count(self) -> int:
        return self._event_count
