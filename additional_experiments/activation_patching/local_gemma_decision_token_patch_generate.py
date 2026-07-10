

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_activation_patch_generate import generate
from local_gemma_activation_patch_pilot import boundary_states, neutral_system_recap, text_layers
from local_gemma_follow_bypass_probe import MODEL_ID, OUT_DIR, build_gemma_prompt, load_case


COMMON_FINAL_PREFIX = (
    "This is a system of three linear equations. We will use the elimination method to solve for "
    "$x$, $y$, and $z$.\n\n"
    "The system of equations is:\n"
    "1) $2x - y + z = 1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="nat_lin3_004::full_cot_normal")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "activation_patching")
    args = parser.parse_args()

    packet, label = load_case(args.case)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    layers = text_layers(model)

    prompt = build_gemma_prompt(processor, packet["original_question"])
    q_prefix = prompt + neutral_system_recap(packet, use_qstar=False) + COMMON_FINAL_PREFIX
    qstar_prefix = prompt + neutral_system_recap(packet, use_qstar=True) + COMMON_FINAL_PREFIX

    q_ids = processor(text=q_prefix, return_tensors="pt")["input_ids"]
    qstar_ids = processor(text=qstar_prefix, return_tensors="pt")["input_ids"]
    if q_ids.shape[-1] != qstar_ids.shape[-1]:
        raise ValueError(f"Decision-prefix lengths differ: Q={q_ids.shape[-1]}, Q*={qstar_ids.shape[-1]}.")
    q_last = processor.tokenizer.decode([int(q_ids[0, -1])], clean_up_tokenization_spaces=False)
    qstar_last = processor.tokenizer.decode([int(qstar_ids[0, -1])], clean_up_tokenization_spaces=False)
    if q_last != "1" or qstar_last != "1":
        raise ValueError(f"Expected both decision prefixes to end in token '1', got {q_last!r} and {qstar_last!r}.")

    print("Capturing the source and target activations for the shared final-answer token '1'.", flush=True)
    q_states = boundary_states(model, processor, layers, q_prefix, span_length=1)
    qstar_states = boundary_states(model, processor, layers, qstar_prefix, span_length=1)

    conditions = [
        ("q_unpatched_after_1", q_prefix, None),
        ("qstar_unpatched_after_1", qstar_prefix, None),
        ("q_after_1_patched_with_qstar", q_prefix, qstar_states),
        ("qstar_after_1_patched_with_q", qstar_prefix, q_states),
    ]
    results = {}
    for name, prefix, source in conditions:
        print(f"Generating: {name}.", flush=True)
        results[name] = generate(
            model,
            processor,
            layers,
            prefix,
            args.max_new_tokens,
            patch_start_layer=0 if source is not None else None,
            source_states=source,
            patch_span_length=1 if source is not None else None,
        )

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "case_id": args.case,
        "manual_label": (label or {}).get("manual_label"),
        "intervention": "replace only the shared final-answer token '1' activation at every decoder-layer input during prefill",
        "prefill_token_count": int(q_ids.shape[-1]),
        "last_prefilled_token": q_last,
        "expected_unpatched_next_tokens": {"q": "2", "qstar": "3"},
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "common_final_prefix": COMMON_FINAL_PREFIX,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.case.replace('::', '__')}__decision_token_1_patch_{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
