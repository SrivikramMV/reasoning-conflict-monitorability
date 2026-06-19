from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .activation_capture import CapturedRun


def token_table(run: CapturedRun) -> pd.DataFrame:
    rows = []
    for i, (tid, tok) in enumerate(zip(run.token_ids, run.tokens)):
        rows.append(
            {
                "position": i,
                "token_id": tid,
                "token": tok,
                "region": "prefilled_context" if i < run.input_token_count else "generated",
            }
        )
    return pd.DataFrame(rows)


def lightweight_run_json(run: CapturedRun) -> dict:
    return {
        "label": run.label,
        "context_text": run.context_text,
        "generated_text": run.generated_text,
        "full_text": run.full_text,
        "input_token_count": run.input_token_count,
        "token_ids": run.token_ids,
        "tokens": run.tokens,
        "metadata": run.metadata,
        "layer_count": run.layer_count,
        "seq_len": run.seq_len,
    }


def save_lightweight_run(run: CapturedRun, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lightweight_run_json(run), indent=2), encoding="utf-8")
