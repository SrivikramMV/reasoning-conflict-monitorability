




from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_boundary_state_batch import changed_line_opening, continuation_logprob
from local_gemma_follow_bypass_probe import MODEL_ID, OUT_DIR, build_gemma_prompt, load_case, strip_answer_channel


def make_depth_prefix(processor: Any, injected_full: str, fraction: float) -> str:
    thought = strip_answer_channel(injected_full)
    ids = processor.tokenizer(thought, add_special_tokens=False)["input_ids"]
    keep = max(1, min(len(ids), round(len(ids) * fraction)))
    text = processor.decode(ids[:keep], skip_special_tokens=False)
    return text.rstrip() + "<channel|>"


def score_depth(model: Any, processor: Any, case_id: str, fraction: float) -> dict[str, Any]:
    packet, label = load_case(case_id)
    injected = make_depth_prefix(processor, packet["injected_cot_full"], fraction)
    boundary_prefix = build_gemma_prompt(processor, packet["original_question"]) + injected
    changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
    q_preamble, q_changed = changed_line_opening(packet["original_question"], changed_line_number)
    qstar_preamble, qstar_changed = changed_line_opening(packet["counterfactual_question"], changed_line_number)
    q_score = continuation_logprob(model, processor, boundary_prefix + q_preamble, q_changed)
    qstar_score = continuation_logprob(model, processor, boundary_prefix + qstar_preamble, qstar_changed)
    return {
        "case_id": case_id,
        "manual_label": (label or {}).get("manual_label"),
        "manual_subtype": (label or {}).get("manual_subtype"),
        "fraction": fraction,
        "delta_avg_qstar_minus_q": qstar_score["avg_logprob"] - q_score["avg_logprob"],
        "q_avg_logprob": q_score["avg_logprob"],
        "qstar_avg_logprob": qstar_score["avg_logprob"],
        "prefix_tail": injected[-700:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["nat_lin3_004::full_cot_normal", "nat_lin3_005::full_cot_normal"])
    parser.add_argument("--fractions", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "prefix_depth_scan")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in args.cases:
        print(f"Depth scan {case_id}", flush=True)
        for fraction in args.fractions:
            row = score_depth(model, processor, case_id, fraction)
            rows.append(row)
            print(f"  {fraction:0.2f}: delta={row['delta_avg_qstar_minus_q']:+.4f}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"prefix_depth_scan_{stamp}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
