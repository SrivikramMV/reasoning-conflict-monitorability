

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ints(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Ignoring incomplete JSONL line {line_number} in {path}", flush=True)
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


@dataclass
class RunPaths:
    bundle_root: Path
    local_root: Path
    drive_root: Path

    def __post_init__(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.drive_root.mkdir(parents=True, exist_ok=True)

    @property
    def local_checkpoints(self) -> Path:
        path = self.local_root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def drive_checkpoints(self) -> Path:
        path = self.drive_root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def analysis_dir(self) -> Path:
        path = self.drive_root / "analysis"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def feature_dir(self) -> Path:
        path = self.drive_root / "hidden_features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def manifest_dir(self) -> Path:
        path = self.drive_root / "manifests"
        path.mkdir(parents=True, exist_ok=True)
        return path


class JsonlCheckpoint:


    def __init__(
        self,
        paths: RunPaths,
        name: str,
        key_field: str,
        checkpoint_every: int = 5,
    ) -> None:
        self.name = name
        self.key_field = key_field
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.local_path = paths.local_checkpoints / f"{name}.jsonl"
        self.drive_path = paths.drive_checkpoints / f"{name}.jsonl"
        if self.drive_path.exists():
            atomic_copy(self.drive_path, self.local_path)
        rows = read_jsonl(self.local_path)
        self.rows: list[dict[str, Any]] = rows
        self.by_key = {str(row[key_field]): row for row in rows}
        if len(self.by_key) != len(rows):
            raise ValueError(f"Duplicate {key_field} values in {self.local_path}")
        if rows:
            write_jsonl(self.local_path, rows)
        self.unsynced = 0

    def __len__(self) -> int:
        return len(self.rows)

    def has(self, key: str) -> bool:
        return str(key) in self.by_key

    def get(self, key: str) -> dict[str, Any] | None:
        return self.by_key.get(str(key))

    def append(self, row: dict[str, Any]) -> None:
        key = str(row[self.key_field])
        if key in self.by_key:
            return
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        self.by_key[key] = row
        self.unsynced += 1
        if self.unsynced >= self.checkpoint_every:
            self.sync()

    def sync(self, force: bool = False) -> None:
        if not self.local_path.exists():
            if not force:
                return
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            self.local_path.touch()
        if self.unsynced or force or not self.drive_path.exists():
            atomic_copy(self.local_path, self.drive_path)
            self.unsynced = 0

    def complete(self, paths: RunPaths, details: dict[str, Any] | None = None) -> None:
        self.sync(force=True)
        marker = {
            "stage": self.name,
            "completed_at_utc": utc_timestamp(),
            "row_count": len(self.rows),
            "checkpoint_sha256": sha256_file(self.drive_path),
            **(details or {}),
        }
        write_json(paths.manifest_dir / f"{self.name}.complete.json", marker)


class Progress:
    def __init__(self, stage: str, total: int, already_done: int = 0) -> None:
        self.stage = stage
        self.total = total
        self.done = already_done
        self.initial_done = already_done
        self.started = time.time()

    def update(self, label: str = "") -> None:
        self.done += 1
        elapsed = max(time.time() - self.started, 1e-6)
        processed = max(self.done - self.initial_done, 1)
        rate = processed / elapsed
        remaining = max(self.total - self.done, 0) / max(rate, 1e-9)
        suffix = f" | {label}" if label else ""
        print(
            f"[{self.stage}] {self.done}/{self.total} | elapsed {elapsed / 60:.1f} min | "
            f"rough remaining {remaining / 60:.1f} min{suffix}",
            flush=True,
        )
