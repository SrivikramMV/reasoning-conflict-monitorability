

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from common import atomic_write_json, utc_now


class RunLogger:
    def __init__(self, output_root: Path, heartbeat_seconds: int, total_rows: int) -> None:
        self.log_dir = output_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "run.log"
        self.progress_path = self.log_dir / "progress.json"
        self.heartbeat_path = self.log_dir / "heartbeat.json"
        self.error_path = self.log_dir / "errors.jsonl"
        self.heartbeat_seconds = heartbeat_seconds
        self.total_rows = total_rows
        self.started = time.time()
        self.stage_started = self.started
        self.initial_global_completed = 0
        self.stage_initial_completed = 0
        self.stage_generated_tokens = 0
        self.state: dict[str, Any] = {
            "status": "initialising",
            "model": None,
            "stage": None,
            "stage_completed": 0,
            "stage_total": 0,
            "global_completed": 0,
            "global_total": total_rows,
            "last_row_id": None,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def log(self, message: str, level: str = "INFO") -> None:
        line = f"{utc_now()} [{level}] {message}"
        print(line, flush=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def begin_stage(self, model: str, stage: str, completed: int, total: int, global_completed: int) -> None:
        with self._lock:
            self.stage_started = time.time()
            self.stage_initial_completed = completed
            self.stage_generated_tokens = 0
            if self.state["status"] == "initialising":
                self.initial_global_completed = global_completed
            self.state.update(
                {
                    "status": "running",
                    "model": model,
                    "stage": stage,
                    "stage_completed": completed,
                    "stage_total": total,
                    "global_completed": global_completed,
                    "last_row_id": None,
                    "last_batch_rows": None,
                    "last_batch_generated_tokens": None,
                    "last_batch_seconds": None,
                    "last_batch_output_tokens_per_second": None,
                }
            )
        self.log(f"Starting {model} / {stage}: {completed}/{total} rows already present")
        self._write_progress()

    def progress(
        self,
        stage_completed: int,
        global_completed: int,
        last_row_id: str | None,
        *,
        batch_rows: int | None = None,
        batch_generated_tokens: int | None = None,
        batch_elapsed_seconds: float | None = None,
    ) -> None:
        with self._lock:
            self.state["stage_completed"] = stage_completed
            self.state["global_completed"] = global_completed
            self.state["last_row_id"] = last_row_id
            if batch_generated_tokens is not None:
                self.stage_generated_tokens += int(batch_generated_tokens)
            if batch_rows is not None:
                self.state["last_batch_rows"] = int(batch_rows)
            if batch_generated_tokens is not None:
                self.state["last_batch_generated_tokens"] = int(batch_generated_tokens)
            if batch_elapsed_seconds is not None:
                elapsed = max(float(batch_elapsed_seconds), 1e-9)
                self.state["last_batch_seconds"] = elapsed
                self.state["last_batch_output_tokens_per_second"] = (
                    int(batch_generated_tokens or 0) / elapsed
                )
        self._write_progress()

    def finish_stage(self) -> None:
        with self._lock:
            model, stage = self.state.get("model"), self.state.get("stage")
            completed, total = self.state.get("stage_completed"), self.state.get("stage_total")
        self.log(f"Completed {model} / {stage}: {completed}/{total}")
        self._write_progress()

    def record_error(self, exc: BaseException, context: dict[str, Any]) -> None:
        record = {
            "created_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context,
        }
        with self._lock:
            with self.error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.state["status"] = "failed"
        self.log(f"{type(exc).__name__}: {exc}", "ERROR")
        self._write_progress()

    def close(self, status: str = "complete") -> None:
        with self._lock:
            self.state["status"] = status
        self._write_progress()
        self._stop.set()
        self._thread.join(timeout=2)

    def _snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            state = dict(self.state)
            stage_started = self.stage_started
        stage_elapsed = max(0.0, now - stage_started)
        overall_elapsed = max(0.0, now - self.started)
        stage_done = int(state.get("stage_completed") or 0)
        stage_total = int(state.get("stage_total") or 0)
        global_done = int(state.get("global_completed") or 0)
        stage_new = max(0, stage_done - self.stage_initial_completed)
        global_new = max(0, global_done - self.initial_global_completed)
        stage_rate = stage_new / stage_elapsed if stage_new and stage_elapsed else None
        overall_rate = global_new / overall_elapsed if global_new and overall_elapsed else None
        stage_eta = (stage_total - stage_done) / stage_rate if stage_rate else None
        overall_eta = (self.total_rows - global_done) / overall_rate if overall_rate else None
        state.update(
            {
                "updated_at_utc": utc_now(),
                "process_id": os.getpid(),
                "stage_elapsed_seconds": stage_elapsed,
                "overall_elapsed_seconds": overall_elapsed,
                "stage_rows_per_second": stage_rate,
                "overall_rows_per_second": overall_rate,
                "stage_eta_seconds": stage_eta,
                "overall_eta_seconds": overall_eta,
                "stage_generated_tokens": self.stage_generated_tokens,
                "stage_output_tokens_per_second": (
                    self.stage_generated_tokens / stage_elapsed
                    if self.stage_generated_tokens and stage_elapsed
                    else None
                ),
                "eta_is_approximate": True,
            }
        )
        return state

    def _write_progress(self) -> None:
        atomic_write_json(self.progress_path, self._snapshot())

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                snapshot = self._snapshot()
                atomic_write_json(self.heartbeat_path, snapshot)
                atomic_write_json(self.progress_path, snapshot)
            except Exception:
                pass
