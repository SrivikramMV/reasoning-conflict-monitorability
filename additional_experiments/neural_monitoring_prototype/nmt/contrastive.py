from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .activation_capture import CapturedRun, pooled_hidden


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def similarity_by_layer(
    current: CapturedRun,
    reference: CapturedRun,
    current_span: Tuple[int, int],
    reference_span: Tuple[int, int],
) -> list[dict]:
    rows = []
    max_layers = min(current.layer_count, reference.layer_count)
    for layer in range(max_layers):
        cur = pooled_hidden(current, layer, current_span[0], current_span[1])
        ref = pooled_hidden(reference, layer, reference_span[0], reference_span[1])
        rows.append({"layer": layer, "similarity": cosine(cur, ref)})
    return rows


def q_vs_qstar_similarity(
    current: CapturedRun,
    q_run: CapturedRun,
    qstar_run: CapturedRun,
    current_span: Tuple[int, int],
    q_span: Tuple[int, int],
    qstar_span: Tuple[int, int],
) -> list[dict]:
    rows = []
    max_layers = min(current.layer_count, q_run.layer_count, qstar_run.layer_count)
    for layer in range(max_layers):
        cur = pooled_hidden(current, layer, current_span[0], current_span[1])
        q = pooled_hidden(q_run, layer, q_span[0], q_span[1])
        qs = pooled_hidden(qstar_run, layer, qstar_span[0], qstar_span[1])
        q_sim = cosine(cur, q)
        qs_sim = cosine(cur, qs)
        rows.append(
            {
                "layer": layer,
                "similarity_to_Q": q_sim,
                "similarity_to_Qstar": qs_sim,
                "Qstar_minus_Q": qs_sim - q_sim,
                "closer_to": "Q*" if qs_sim > q_sim else "Q",
            }
        )
    return rows


def nearest_runs(
    current: CapturedRun,
    references: Dict[str, CapturedRun],
    current_span: Tuple[int, int],
    reference_span_mode: str = "generated",
    layer: int | None = None,
    top_n: int = 8,
) -> list[dict]:
    if layer is None:
        layer = current.layer_count - 1
    cur = pooled_hidden(current, layer, current_span[0], current_span[1])
    rows = []
    for label, ref in references.items():
        if label == current.label:
            continue
        if reference_span_mode == "generated":
            start, end = ref.input_token_count, ref.seq_len
        elif reference_span_mode == "full":
            start, end = 0, ref.seq_len
        else:
            start, end = max(0, ref.seq_len - 16), ref.seq_len
        rows.append(
            {
                "reference": label,
                "similarity": cosine(cur, pooled_hidden(ref, min(layer, ref.layer_count - 1), start, end)),
                "span": reference_span_mode,
            }
        )
    return sorted(rows, key=lambda x: x["similarity"], reverse=True)[:top_n]
