













from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    OUT_DIR,
    build_gemma_prompt,
    gemma_stop_token_ids,
    load_case,
)


def extract_thought_prefix(raw_text: str) -> str:
    if "<channel|>" in raw_text:
        return raw_text.split("<channel|>", 1)[0] + "<channel|>"
    match = re.search(r"(<\|channel\>thought\n.*?)(?:<channel\|>)", raw_text, re.DOTALL)
    if match:
        return match.group(1) + "<channel|>"
    raise ValueError("Could not extract thought prefix ending in <channel|>.")


def generate_clean_prefix(model: Any, processor: Any, question: str, max_new_tokens: int) -> dict[str, Any]:
    prompt = build_gemma_prompt(processor, question)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=gemma_stop_token_ids(processor),
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    continuation = processor.decode(outputs[0][input_len:].tolist(), skip_special_tokens=False)
    prefix = extract_thought_prefix(continuation)
    return {
        "prompt": prompt,
        "continuation": continuation,
        "thought_prefix": prefix,
    }


def boundary_hidden_states(model: Any, processor: Any, text: str) -> list[torch.Tensor]:
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return [h[0, -1, :].float().detach().cpu() for h in out.hidden_states]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def run_case(model: Any, processor: Any, case_id: str, max_new_tokens: int) -> dict[str, Any]:
    packet, label = load_case(case_id)
    clean_q = generate_clean_prefix(model, processor, packet["original_question"], max_new_tokens=max_new_tokens)

    q_text = clean_q["prompt"] + clean_q["thought_prefix"]
    intervention_text = build_gemma_prompt(processor, packet["original_question"]) + packet["injected_cot_full"]
    qstar_text = build_gemma_prompt(processor, packet["counterfactual_question"]) + packet["injected_cot_full"]

    h_q = boundary_hidden_states(model, processor, q_text)
    h_i = boundary_hidden_states(model, processor, intervention_text)
    h_qstar = boundary_hidden_states(model, processor, qstar_text)

    rows = []
    for layer, (q, i, qs) in enumerate(zip(h_q, h_i, h_qstar)):
        sim_q = cosine(i, q)
        sim_qstar = cosine(i, qs)
        rows.append(
            {
                "layer": layer,
                "cos_intervention_to_aligned_q": sim_q,
                "cos_intervention_to_aligned_qstar": sim_qstar,
                "delta_qstar_minus_q": sim_qstar - sim_q,
            }
        )

    return {
        "case_id": case_id,
        "manual_label": (label or {}).get("manual_label"),
        "manual_subtype": (label or {}).get("manual_subtype"),
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "clean_q_thought_prefix_tail": clean_q["thought_prefix"][-1200:],
        "layer_similarity": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["nat_lin3_004::full_cot_normal", "nat_lin3_005::full_cot_normal"])
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "hidden_state_probe")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case_id in args.cases:
        print(f"Running hidden-state probe for {case_id}", flush=True)
        row = run_case(model, processor, case_id, args.max_new_tokens)
        results.append(row)
        early = row["layer_similarity"][1]["delta_qstar_minus_q"]
        mid = row["layer_similarity"][len(row["layer_similarity"]) // 2]["delta_qstar_minus_q"]
        late = row["layer_similarity"][-1]["delta_qstar_minus_q"]
        print(f"  delta qstar-q: early={early:+.4f}, mid={mid:+.4f}, late={late:+.4f}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"hidden_state_probe_{stamp}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
