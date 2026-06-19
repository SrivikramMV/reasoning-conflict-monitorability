from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .activation_capture import CapturedRun


@dataclass
class LensToken:
    token: str
    token_id: int
    score: float
    probability: float


def _final_norm(model, hidden: torch.Tensor) -> torch.Tensor:
                                                                              
                                               
    norm = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
    if norm is None:
        return hidden
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    return norm(hidden.to(device=device, dtype=dtype)).detach().cpu()


def _lm_head(model, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        weight = getattr(model.lm_head, "weight", None)
        if weight is not None:
            h = hidden.to(device=weight.device, dtype=weight.dtype)
        else:
            param = next(model.parameters())
            h = hidden.to(device=param.device, dtype=param.dtype)
        return model.lm_head(h).detach().cpu()
    embeddings = model.get_output_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None:
        h = hidden.to(device=weight.device, dtype=weight.dtype)
    else:
        param = next(model.parameters())
        h = hidden.to(device=param.device, dtype=param.dtype)
    return embeddings(h).detach().cpu()


def layer_logits(model, hidden: torch.Tensor, apply_final_norm: bool = True) -> torch.Tensor:
    h = _final_norm(model, hidden) if apply_final_norm else hidden
    return _lm_head(model, h)


def top_tokens_for_state(
    model,
    tokenizer,
    hidden: torch.Tensor,
    top_k: int = 10,
    apply_final_norm: bool = True,
) -> List[LensToken]:
    logits = layer_logits(model, hidden, apply_final_norm=apply_final_norm)
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(logits, k=top_k, dim=-1)
    out: List[LensToken] = []
    for score, tid in zip(vals.tolist(), idxs.tolist()):
        token = tokenizer.decode([int(tid)], skip_special_tokens=False).replace("\n", "\\n")
        out.append(
            LensToken(
                token=token,
                token_id=int(tid),
                score=float(score),
                probability=float(probs[int(tid)].item()),
            )
        )
    return out


def top_tokens_grid(
    model,
    tokenizer,
    run: CapturedRun,
    token_position: int,
    top_k: int = 8,
    apply_final_norm: bool = True,
) -> list[dict]:
    token_position = max(0, min(token_position, run.seq_len - 1))
    rows = []
    for layer, states in enumerate(run.hidden_states_cpu):
        toks = top_tokens_for_state(
            model,
            tokenizer,
            states[token_position],
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )
        rows.append(
            {
                "layer": layer,
                "decoded_top_tokens": ", ".join(t.token for t in toks),
                "top_token": toks[0].token if toks else "",
                "top_score": toks[0].score if toks else 0.0,
                "top_probability": toks[0].probability if toks else 0.0,
            }
        )
    return rows


def layer_transcript(
    model,
    tokenizer,
    run: CapturedRun,
    layer: int,
    start: int,
    end: int,
    apply_final_norm: bool = True,
) -> str:
    layer = max(0, min(layer, run.layer_count - 1))
    start = max(0, min(start, run.seq_len - 1))
    end = max(start + 1, min(end, run.seq_len))
    pieces = []
    for pos in range(start, end):
        toks = top_tokens_for_state(
            model,
            tokenizer,
            run.hidden_states_cpu[layer][pos],
            top_k=1,
            apply_final_norm=apply_final_norm,
        )
        if toks:
            pieces.append(toks[0].token)
    return "".join(pieces).replace("▁", " ")


def transcript_by_layer(
    model,
    tokenizer,
    run: CapturedRun,
    start: int,
    end: int,
    stride: int = 2,
    apply_final_norm: bool = True,
) -> list[dict]:
    rows = []
    for layer in range(0, run.layer_count, max(1, stride)):
        rows.append(
            {
                "layer": layer,
                "approx_layer_transcript": layer_transcript(
                    model,
                    tokenizer,
                    run,
                    layer,
                    start,
                    end,
                    apply_final_norm=apply_final_norm,
                ),
            }
        )
    if rows and rows[-1]["layer"] != run.layer_count - 1:
        rows.append(
            {
                "layer": run.layer_count - 1,
                "approx_layer_transcript": layer_transcript(
                    model,
                    tokenizer,
                    run,
                    run.layer_count - 1,
                    start,
                    end,
                    apply_final_norm=apply_final_norm,
                ),
            }
        )
    return rows
