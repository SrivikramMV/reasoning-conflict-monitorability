








from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_boundary_state_batch import continuation_logprob, pretty_equation, question_lines
from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    ROOT,
    build_gemma_prompt,
    gemma_stop_token_ids,
    load_case,
)
from local_gemma_heldout_boundary_extract import cohort, completed_cot


DEFAULT_CASES = [
    "nat_lin3_004::full_cot_normal",                    
    "nat_lin3_013::full_cot_normal",                    
    "nat_lin3_003::full_cot_normal",                        
    "nat_lin3_010::full_cot_normal",                        
    "nat_lin3_008::full_cot_normal",                      
    "nat_lin3_009::full_cot_normal",                      
    "nat_lin3_005::full_cot_normal",                      
    "nat_lin3_011::full_cot_normal",                      
]

TEMPLATES = {
    "standard_system": (
        "This is a system of three linear equations. We will use the elimination "
        "method to solve for $x$, $y$, and $z$.\n\nThe system of equations is:\n"
    ),
    "plain_restatement": "The equations are:\n",
    "active_system": "I will solve the following active system:\n",
    "verification": "Checking the problem statement, the equations are:\n",
    "minimal": "",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_pair(packet: dict[str, Any], template: str) -> tuple[str, str, str]:
    changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
    idx = max(0, changed_line_number - 2)
    q_lines = question_lines(packet["original_question"])
    qstar_lines = question_lines(packet["counterfactual_question"])
    if len(q_lines) != len(qstar_lines) or idx >= len(q_lines):
        raise ValueError(f"Cannot align changed equation for {packet['intervention_id']}")

    preamble = TEMPLATES[template]
    for line_idx, equation in enumerate(q_lines[:idx], start=1):
        preamble += f"{line_idx}) ${pretty_equation(equation)}$\n"

    number = idx + 1
    if template == "minimal":
        q_target = f"{number}) ${pretty_equation(q_lines[idx])}$\n"
        qstar_target = f"{number}) ${pretty_equation(qstar_lines[idx])}$\n"
    else:
        q_target = f"{number}) ${pretty_equation(q_lines[idx])}$\n"
        qstar_target = f"{number}) ${pretty_equation(qstar_lines[idx])}$\n"
    return preamble, q_target, qstar_target


def normalise_equation(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("$", "").lower()


def classify_generated(text: str, q_target: str, qstar_target: str) -> str:
    head = normalise_equation(text[:800])
    q = normalise_equation(q_target)
    qstar = normalise_equation(qstar_target)
    if qstar in head:
        return "QSTAR"
    if q in head:
        return "Q"
    return "OTHER"


def generate_after_preamble(
    model: Any,
    processor: Any,
    prefix: str,
    max_new_tokens: int,
) -> str:
    inputs = processor(text=prefix, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=gemma_stop_token_ids(processor),
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    continuation = processor.tokenizer.decode(
        output[0, input_len:], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    del inputs, output
    torch.cuda.empty_cache()
    return continuation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--all-cohort", action="store_true")
    parser.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "candidate_template_robustness",
    )
    args = parser.parse_args()

    if args.all_cohort:
        args.cases = [packet["intervention_id"] for packet, _ in cohort()]

    unknown = set(args.templates) - set(TEMPLATES)
    if unknown:
        raise ValueError(f"Unknown templates: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        packet, label = load_case(case_id)
        boundary = build_gemma_prompt(processor, packet["original_question"]) + completed_cot(packet)
        for template in args.templates:
            print(f"Scoring {case_id} / {template}", flush=True)
            preamble, q_target, qstar_target = candidate_pair(packet, template)
            prefix = boundary + preamble
            q_score = continuation_logprob(model, processor, prefix, q_target)
            qstar_score = continuation_logprob(model, processor, prefix, qstar_target)
            delta_avg = qstar_score["avg_logprob"] - q_score["avg_logprob"]
            delta_total = qstar_score["total_logprob"] - q_score["total_logprob"]
            generated = ""
            generated_state = "NOT_RUN"
            if args.generate:
                generated = generate_after_preamble(model, processor, prefix, args.max_new_tokens)
                generated_state = classify_generated(generated, q_target, qstar_target)
            row = {
                "created_at_utc": utc_now(),
                "model_id": MODEL_ID,
                "case_id": case_id,
                "item_id": case_id.split("::", 1)[0],
                "variant": case_id.split("::", 1)[1],
                "manual_label": (label or {}).get("manual_label"),
                "template": template,
                "preamble": preamble,
                "q_target": q_target,
                "qstar_target": qstar_target,
                "q": q_score,
                "qstar": qstar_score,
                "delta_avg_qstar_minus_q": delta_avg,
                "delta_total_qstar_minus_q": delta_total,
                "preference": "QSTAR" if delta_avg > 0 else "Q",
                "generated_state": generated_state,
                "generated": generated,
            }
            rows.append(row)
            print(
                f"  delta={delta_avg:+.5f}; preference={row['preference']}; "
                f"generated={generated_state}",
                flush=True,
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_dir / f"candidate_template_smoke_{stamp}.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    out_jsonl = out_json.with_suffix(".jsonl")
    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
