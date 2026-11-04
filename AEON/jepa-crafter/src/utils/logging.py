"""Structured logging utilities for metrics, diagnostics, and experiments."""
import os
import json
import csv
from typing import Dict, Any, Optional
import time


class MetricLogger:
    """Logs structured metrics to JSONL and optional CSV files."""
    
    def __init__(self, log_dir: str, prefix: str = "metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, f"{prefix}.jsonl")
        self.csv_path = os.path.join(log_dir, f"{prefix}.csv")
        self._csv_writer = None
        self._csv_file = None
        self._fieldnames = None

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log a dictionary of metrics at a specific step."""
        record = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        # Update CSV
        flat_record = {}
        for k, v in record.items():
            if isinstance(v, (int, float, str, bool)):
                flat_record[k] = v
            elif isinstance(v, (list, tuple)) and len(v) <= 4:
                flat_record[k] = str(v)
                
        if self._fieldnames is None:
            self._fieldnames = list(flat_record.keys())
            write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                self._csv_writer.writeheader()
        
        if self._csv_writer is not None:
            self._csv_writer.writerow(flat_record)
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()


def save_json(data: Any, path: str) -> None:
    """Save an object to JSON with indentation."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load an object from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
