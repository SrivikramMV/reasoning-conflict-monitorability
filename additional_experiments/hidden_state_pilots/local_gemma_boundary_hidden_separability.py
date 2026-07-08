








from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_boundary_state_batch import changed_line_opening, continuation_logprob
from local_gemma_follow_bypass_probe import MODEL_ID, OUT_DIR, build_gemma_prompt, load_case


DEFAULT_CASES = [
    "nat_lin3_002::full_cot_normal",          
    "nat_lin3_003::full_cot_normal",          
    "nat_lin3_004::full_cot_normal",          
    "nat_lin3_008::full_cot_normal",          
    "nat_lin3_005::full_cot_normal",          
    "nat_lin3_006::full_cot_normal",          
    "nat_lin3_007::full_cot_normal",          
    "nat_lin3_010::full_cot_normal",          
]


LABEL_MAP = {
    "FOLLOW": "follow",
    "SILENT_BYPASS_REANCHOR": "bypass",
}


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def boundary_hidden_stack(model: Any, processor: Any, text: str) -> np.ndarray:
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
                                                                      
    stack = np.stack([h[0, -1, :].float().detach().cpu().numpy() for h in out.hidden_states], axis=0)
    del out
    del inputs
    torch.cuda.empty_cache()
    return stack


def boundary_delta_score(model: Any, processor: Any, packet: dict[str, Any], boundary_text: str) -> float:
    changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
    q_preamble, q_changed = changed_line_opening(packet["original_question"], changed_line_number)
    qstar_preamble, qstar_changed = changed_line_opening(packet["counterfactual_question"], changed_line_number)
    q = continuation_logprob(model, processor, boundary_text + q_preamble, q_changed)
    qstar = continuation_logprob(model, processor, boundary_text + qstar_preamble, qstar_changed)
    return qstar["avg_logprob"] - q["avg_logprob"]


def extract_case(model: Any, processor: Any, case_id: str) -> dict[str, Any]:
    packet, label_row = load_case(case_id)
    manual_label = (label_row or {}).get("manual_label")
    if manual_label not in LABEL_MAP:
        raise ValueError(f"Case {case_id} has unsupported label {manual_label}")
    boundary_text = build_gemma_prompt(processor, packet["original_question"]) + packet["injected_cot_full"]
    hidden = boundary_hidden_stack(model, processor, boundary_text)
    delta = boundary_delta_score(model, processor, packet, boundary_text)
    return {
        "case_id": case_id,
        "class": LABEL_MAP[manual_label],
        "manual_label": manual_label,
        "manual_subtype": (label_row or {}).get("manual_subtype"),
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "changed_line": packet.get("changed_line"),
        "boundary_delta_qstar_minus_q": delta,
        "cot_tail": packet["injected_cot_full"][-900:],
        "hidden": hidden,
    }


def leave_one_out_centroid_accuracy(hidden: np.ndarray, y: np.ndarray) -> dict[str, Any]:




    n, layers, _ = hidden.shape
    hidden_norm = l2_normalize(hidden)
    rows = []
    for layer in range(layers):
        preds = []
        margins = []
        for i in range(n):
            train = np.arange(n) != i
            follow_centroid = hidden_norm[train & (y == 1), layer, :].mean(axis=0)
            bypass_centroid = hidden_norm[train & (y == 0), layer, :].mean(axis=0)
            follow_centroid = follow_centroid / (np.linalg.norm(follow_centroid) + 1e-9)
            bypass_centroid = bypass_centroid / (np.linalg.norm(bypass_centroid) + 1e-9)
            vec = hidden_norm[i, layer, :]
            sim_follow = float(vec @ follow_centroid)
            sim_bypass = float(vec @ bypass_centroid)
            pred = 1 if sim_follow > sim_bypass else 0
            preds.append(pred)
            margins.append(sim_follow - sim_bypass)
        acc = float(np.mean(np.array(preds) == y))
        rows.append(
            {
                "layer": layer,
                "loo_accuracy": acc,
                "predictions": preds,
                "margins_follow_minus_bypass": margins,
            }
        )
    return {"layers": rows}


def summarize_best_layers(stats: dict[str, Any], labels: list[str], case_ids: list[str], y: np.ndarray) -> list[dict[str, Any]]:
    best = sorted(stats["layers"], key=lambda r: (r["loo_accuracy"], r["layer"]), reverse=True)[:8]
    out = []
    for row in best:
        pred_labels = ["follow" if p == 1 else "bypass" for p in row["predictions"]]
        mistakes = [
            {"case_id": cid, "true": true, "pred": pred}
            for cid, true, pred in zip(case_ids, labels, pred_labels)
            if true != pred
        ]
        out.append(
            {
                "layer": row["layer"],
                "loo_accuracy": row["loo_accuracy"],
                "mistakes": mistakes,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "hidden_separability")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    hidden_rows = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_jsonl = args.out_dir / f"hidden_separability_partial_{stamp}.jsonl"
    partial_npz = args.out_dir / f"hidden_separability_partial_{stamp}.npz"
    for case_id in args.cases:
        print(f"Extracting {case_id}", flush=True)
        row = extract_case(model, processor, case_id)
        hidden_rows.append(row.pop("hidden"))
        extracted.append(row)
        with partial_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.savez_compressed(
            partial_npz,
            hidden=np.stack(hidden_rows, axis=0),
            case_ids=np.array([r["case_id"] for r in extracted]),
            labels=np.array([r["class"] for r in extracted]),
        )
        torch.cuda.empty_cache()
        print(
            f"  class={row['class']} boundary_delta={row['boundary_delta_qstar_minus_q']:+.4f}",
            flush=True,
        )

    hidden = np.stack(hidden_rows, axis=0)
    labels = [r["class"] for r in extracted]
    y = np.array([1 if label == "follow" else 0 for label in labels], dtype=int)
    stats = leave_one_out_centroid_accuracy(hidden, y)
    best_layers = summarize_best_layers(stats, labels, [r["case_id"] for r in extracted], y)

    payload = {
        "created_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "cases": extracted,
        "hidden_shape": list(hidden.shape),
        "best_layers": best_layers,
        "layer_stats": stats["layers"],
    }
    out_json = args.out_dir / f"hidden_separability_{stamp}.json"
    out_npz = args.out_dir / f"hidden_separability_{stamp}.npz"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_npz, hidden=hidden, y=y, case_ids=np.array([r["case_id"] for r in extracted]))

    print("\nBest layers:")
    for row in best_layers:
        print(f"  layer {row['layer']}: acc={row['loo_accuracy']:.3f}, mistakes={row['mistakes']}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
