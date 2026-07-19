

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_ints(values: Iterable[int]) -> str:
    return sha256_text(",".join(str(int(value)) for value in values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(model_revision: str, row_id: str, seed_value: int) -> int:
    payload = f"{model_revision}|{row_id}|{seed_value}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)
    return value or 1


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class JsonlCheckpoint:
    def __init__(self, path: Path, id_field: str) -> None:
        self.path = path
        self.id_field = id_field
        self.rows = read_jsonl(path)
        self.ids = {str(row[id_field]) for row in self.rows}
        if len(self.ids) != len(self.rows):
            raise ValueError(f"Duplicate {id_field} values in {path}")

    def missing(self, row_id: str) -> bool:
        return row_id not in self.ids

    def append(self, rows: list[dict[str, Any]]) -> None:
        fresh = [row for row in rows if str(row[self.id_field]) not in self.ids]
        append_jsonl(self.path, fresh)
        self.rows.extend(fresh)
        self.ids.update(str(row[self.id_field]) for row in fresh)


def find_subsequence(sequence: list[int], subsequence: list[int], start: int = 0) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("Cannot locate an empty token subsequence")
    for index in range(start, len(sequence) - len(subsequence) + 1):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index, index + len(subsequence)
    raise ValueError("Token subsequence not found")


def token_safe_cut(tokenizer: Any, thought_ids: list[int], ratio: float) -> dict[str, Any]:
    if not 0 < ratio < 1:
        raise ValueError("Partial injection ratio must lie between zero and one")
    count = len(thought_ids)
    if count < 12:
        raise ValueError("Source reasoning is too short for a partial intervention")
    target = max(4, min(count - 2, round(count * ratio)))
    radius = max(8, round(count * 0.12))
    lower, upper = max(4, target - radius), min(count - 1, target + radius)
    candidates = []
    for cut in range(lower, upper + 1):
        decoded = tokenizer.decode(
            thought_ids[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        stripped = decoded.rstrip()
        if re.search(r"(?:[.!?](?:[\"')\]]*)|\n)\s*$", decoded):
            quality = 0
        elif decoded.endswith((" ", "\n", "\t")):
            quality = 1
        elif stripped.endswith((",", ";", ":")):
            quality = 2
        else:
            quality = 3
        candidates.append((quality, abs(cut - target), cut, decoded))
    quality, distance, cut, decoded = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    labels = {
        0: "sentence_or_line_boundary",
        1: "whitespace_boundary",
        2: "clause_boundary",
        3: "token_boundary",
    }
    return {
        "cut_token_count": cut,
        "source_thought_token_count": count,
        "target_ratio": ratio,
        "actual_ratio": cut / count,
        "target_token_count": target,
        "distance_from_target_tokens": distance,
        "boundary_quality": labels[quality],
        "decoded_prefix": decoded,
    }


def complete_cut(tokenizer: Any, thought_ids: list[int]) -> dict[str, Any]:
    return {
        "cut_token_count": len(thought_ids),
        "source_thought_token_count": len(thought_ids),
        "target_ratio": 1.0,
        "actual_ratio": 1.0,
        "target_token_count": len(thought_ids),
        "distance_from_target_tokens": 0,
        "boundary_quality": "complete_trace",
        "decoded_prefix": tokenizer.decode(
            thought_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
    }


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
