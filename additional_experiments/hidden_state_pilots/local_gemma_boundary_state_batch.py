







from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    OUT_DIR,
    build_gemma_prompt,
    generate,
    load_case,
    make_variant,
)


DEFAULT_CASES = [
    "nat_lin3_004::full_cot_normal",
    "nat_lin3_005::full_cot_normal",
    "nat_lin3_002::full_cot_normal",
    "nat_lin3_008::full_cot_normal",
    "nat_lin3_013::cot_50_percent",
    "nat_lin3_013::cot_90_percent",
    "nat_lin3_013::full_cot_normal",
]

DEFAULT_VARIANTS = [
    "original",
    "truncate_before_closure",
    "append_present_solution",
    "append_prompt_check",
    "append_scratchpad_continuation",
    "append_closed_trace",
    "append_qstar_system_recap",
    "append_q_system_recap",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pretty_equation(line: str) -> str:
    line = re.sub(r"\b1([a-z])\b", r"\1", line)
    line = re.sub(r"\+-", "-", line)
    line = re.sub(r"\+\s*-", "- ", line)
    line = re.sub(r"-1([a-z])\b", r"- \1", line)
    line = re.sub(r"\+\s*1([a-z])\b", r"+ \1", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = line.replace("+ -", "- ")
    return line


def question_lines(question: str) -> list[str]:
    return [line.strip() for line in question.splitlines() if "=" in line]


def system_opening(question: str) -> str:
    equations = question_lines(question)
    if len(equations) == 3:
        variables = "$x$, $y$, and $z$"
        heading = "This is a system of three linear equations. We will use the elimination method to solve for $x$, $y$, and $z$."
    elif len(equations) == 2:
        variables = "$x$ and $y$"
        heading = "This is a system of two linear equations. We can solve this using either the substitution method or the elimination method."
    else:
        variables = "the variables"
        heading = f"We need to solve for {variables}."
    lines = [heading, "", "The system of equations is:"]
    for i, eq in enumerate(equations, start=1):
        lines.append(f"{i}) ${pretty_equation(eq)}$")
    return "\n".join(lines)


def changed_line_opening(question: str, changed_line_number: int) -> str:
    equations = question_lines(question)
    idx = max(0, changed_line_number - 2)
    if len(equations) == 3:
        heading = "This is a system of three linear equations. We will use the elimination method to solve for $x$, $y$, and $z$."
    elif len(equations) == 2:
        heading = "This is a system of two linear equations. We can solve this using either the substitution method or the elimination method."
    else:
        heading = "We need to solve the problem."
    preamble = heading + "\n\nThe system of equations is:\n"
    previous = ""
    for i, eq in enumerate(equations[:idx], start=1):
        previous += f"{i}) ${pretty_equation(eq)}$\n"
    target = f"{idx + 1}) ${pretty_equation(equations[idx])}$\n"
    return preamble + previous, target


def continuation_logprob(
    model: Any,
    processor: Any,
    prefix_text: str,
    continuation_text: str,
) -> dict[str, Any]:
    prefix_inputs = processor(text=prefix_text, return_tensors="pt").to(model.device)
    full_inputs = processor(text=prefix_text + continuation_text, return_tensors="pt").to(model.device)
    prefix_len = int(prefix_inputs["input_ids"].shape[-1])
    full_ids = full_inputs["input_ids"]
    target_ids = full_ids[:, prefix_len:]
    if target_ids.numel() == 0:
        raise ValueError("Continuation produced no tokens.")

    with torch.inference_mode():
        out = model(**full_inputs)
        logits = out.logits[:, prefix_len - 1 : -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    token_log_probs_list = token_log_probs[0].detach().cpu().tolist()
    total = float(sum(token_log_probs_list))
    count = len(token_log_probs_list)
    return {
        "text": continuation_text,
        "token_count": count,
        "total_logprob": total,
        "avg_logprob": total / count,
        "perplexity": math.exp(-total / count),
    }


def score_case_variant(model: Any, processor: Any, case_id: str, variant: str, max_new_tokens: int) -> dict[str, Any]:
    packet, label = load_case(case_id)
    injected = make_variant(packet, variant)
    prompt = build_gemma_prompt(processor, packet["original_question"])
    boundary_prefix = prompt + injected

    changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
    q_preamble, q_changed = changed_line_opening(packet["original_question"], changed_line_number)
    qstar_preamble, qstar_changed = changed_line_opening(packet["counterfactual_question"], changed_line_number)

    q_opening = system_opening(packet["original_question"])
    qstar_opening = system_opening(packet["counterfactual_question"])

    full_q = continuation_logprob(model, processor, boundary_prefix, q_opening)
    full_qstar = continuation_logprob(model, processor, boundary_prefix, qstar_opening)
    line_q = continuation_logprob(model, processor, boundary_prefix + q_preamble, q_changed)
    line_qstar = continuation_logprob(model, processor, boundary_prefix + qstar_preamble, qstar_changed)

    generation = generate(
        model=model,
        processor=processor,
        torch=torch,
        question=packet["original_question"],
        injected_raw_prefix=injected,
        max_new_tokens=max_new_tokens,
        inspect_boundary=False,
    )

    return {
        "case_id": case_id,
        "variant": variant,
        "model_id": MODEL_ID,
        "created_at_utc": utc_now_iso(),
        "manual_label": (label or {}).get("manual_label"),
        "manual_subtype": (label or {}).get("manual_subtype"),
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "changed_line": packet.get("changed_line"),
        "scores": {
            "full_opening_q": full_q,
            "full_opening_qstar": full_qstar,
            "changed_line_q": line_q,
            "changed_line_qstar": line_qstar,
            "full_opening_delta_avg_qstar_minus_q": full_qstar["avg_logprob"] - full_q["avg_logprob"],
            "changed_line_delta_avg_qstar_minus_q": line_qstar["avg_logprob"] - line_q["avg_logprob"],
        },
        "generated_final_head": generation["final_answer"][:1200],
        "generated_cot_tail": generation["cot"][-1200:],
        "generated_token_count": generation["generated_token_count"],
        "elapsed_seconds": generation["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--variants", nargs="*", default=["original"])
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--changed-line-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "boundary_state_batch")
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS if args.all_variants else args.variants
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = args.out_dir / f"boundary_state_batch_{stamp}.jsonl"
    for case_id in args.cases:
        for variant in variants:
            print(f"Running {case_id} / {variant}", flush=True)
            if args.score_only:
                packet, label = load_case(case_id)
                injected = make_variant(packet, variant)
                prompt = build_gemma_prompt(processor, packet["original_question"])
                boundary_prefix = prompt + injected
                changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
                q_preamble, q_changed = changed_line_opening(packet["original_question"], changed_line_number)
                qstar_preamble, qstar_changed = changed_line_opening(packet["counterfactual_question"], changed_line_number)
                line_q = continuation_logprob(model, processor, boundary_prefix + q_preamble, q_changed)
                line_qstar = continuation_logprob(model, processor, boundary_prefix + qstar_preamble, qstar_changed)
                if args.changed_line_only:
                    full_q = {"avg_logprob": 0.0, "total_logprob": 0.0, "token_count": 0, "perplexity": 0.0, "text": ""}
                    full_qstar = {"avg_logprob": 0.0, "total_logprob": 0.0, "token_count": 0, "perplexity": 0.0, "text": ""}
                else:
                    q_opening = system_opening(packet["original_question"])
                    qstar_opening = system_opening(packet["counterfactual_question"])
                    full_q = continuation_logprob(model, processor, boundary_prefix, q_opening)
                    full_qstar = continuation_logprob(model, processor, boundary_prefix, qstar_opening)
                row = {
                    "case_id": case_id,
                    "variant": variant,
                    "model_id": MODEL_ID,
                    "created_at_utc": utc_now_iso(),
                    "manual_label": (label or {}).get("manual_label"),
                    "manual_subtype": (label or {}).get("manual_subtype"),
                    "original_question": packet["original_question"],
                    "counterfactual_question": packet["counterfactual_question"],
                    "original_answer": packet["original_answer"],
                    "counterfactual_answer": packet["counterfactual_answer"],
                    "changed_line": packet.get("changed_line"),
                    "scores": {
                        "full_opening_q": full_q,
                        "full_opening_qstar": full_qstar,
                        "changed_line_q": line_q,
                        "changed_line_qstar": line_qstar,
                        "full_opening_delta_avg_qstar_minus_q": full_qstar["avg_logprob"] - full_q["avg_logprob"],
                        "changed_line_delta_avg_qstar_minus_q": line_qstar["avg_logprob"] - line_q["avg_logprob"],
                    },
                    "generated_final_head": "",
                    "generated_cot_tail": "",
                    "generated_token_count": 0,
                    "elapsed_seconds": None,
                }
            else:
                row = score_case_variant(model, processor, case_id, variant, args.max_new_tokens)
            results.append(row)
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            d = row["scores"]["changed_line_delta_avg_qstar_minus_q"]
            f = row["generated_final_head"][:180].replace("\n", " | ")
            print(f"  changed-line delta Q*-Q avg logprob: {d:+.4f}", flush=True)
            if f:
                print(f"  final head: {f}", flush=True)
            print(f"  partial saved: {out_jsonl}", flush=True)

    out_json = args.out_dir / f"boundary_state_batch_{stamp}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = args.out_dir / f"boundary_state_batch_{stamp}.csv"
    lines = [
        "case_id,variant,manual_label,manual_subtype,delta_full_avg_qstar_minus_q,delta_changed_line_avg_qstar_minus_q,generated_final_head"
    ]
    for row in results:
        head = row["generated_final_head"].replace("\n", " ").replace('"', '""')[:400]
        lines.append(
            ",".join(
                [
                    row["case_id"],
                    row["variant"],
                    str(row.get("manual_label") or ""),
                    str(row.get("manual_subtype") or ""),
                    f"{row['scores']['full_opening_delta_avg_qstar_minus_q']:.6f}",
                    f"{row['scores']['changed_line_delta_avg_qstar_minus_q']:.6f}",
                    f'"{head}"',
                ]
            )
        )
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
