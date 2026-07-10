

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_activation_patch_pilot import boundary_states, neutral_system_recap, text_layers
from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    OUT_DIR,
    build_gemma_prompt,
    gemma_stop_token_ids,
    load_case,
    strip_answer_channel,
)


def generate(
    model: Any,
    processor: Any,
    layers: Any,
    boundary_text: str,
    max_new_tokens: int,
    patch_start_layer: int | None = None,
    source_states: list[torch.Tensor] | None = None,
    patch_span_length: int | None = None,
) -> dict[str, Any]:
    inputs = processor(text=boundary_text, return_tensors="pt").to(model.device)
    input_length = int(inputs["input_ids"].shape[-1])
    handles = []
    if patch_start_layer is not None:
        if source_states is None or patch_span_length is None:
            raise ValueError("source_states and patch_span_length are required for patching.")
        for layer_index in range(patch_start_layer, len(layers)):
            patch = source_states[layer_index].to(device=model.device, dtype=model.dtype)

            def replace_prefill_span(
                module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                value: torch.Tensor = patch,
            ):
                hidden = args[0]
                                                                                    
                if hidden.shape[1] == input_length:
                    replaced = hidden.clone()
                    replaced[:, -patch_span_length:, :] = value
                    return (replaced, *args[1:]), kwargs
                return args, kwargs

            handles.append(
                layers[layer_index].register_forward_pre_hook(replace_prefill_span, with_kwargs=True)
            )

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=gemma_stop_token_ids(processor),
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()

    generated_ids = output_ids[:, input_length:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return {
        "text": text,
        "token_count": int(generated_ids.shape[-1]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="nat_lin3_004::full_cot_normal")
    parser.add_argument("--patch-start-layer", type=int, default=10)
    parser.add_argument("--patch-scope", choices=["recap", "boundary"], default="recap")
    parser.add_argument("--max-new-tokens", type=int, default=160)
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
    thought_only = strip_answer_channel(packet["injected_cot_full"]).rstrip()
    unrecapped_boundary = prompt + thought_only
    q_boundary = prompt + neutral_system_recap(packet, use_qstar=False)
    qstar_boundary = prompt + neutral_system_recap(packet, use_qstar=True)

    base_len = int(processor(text=unrecapped_boundary, return_tensors="pt")["input_ids"].shape[-1])
    q_len = int(processor(text=q_boundary, return_tensors="pt")["input_ids"].shape[-1])
    qstar_len = int(processor(text=qstar_boundary, return_tensors="pt")["input_ids"].shape[-1])
    if q_len != qstar_len:
        raise ValueError(f"Matched boundary lengths differ: Q={q_len}, Q*={qstar_len}.")
    recap_span_length = q_len - base_len
    patch_span_length = 1 if args.patch_scope == "boundary" else recap_span_length

    print(f"Capturing aligned {args.patch_scope} states.", flush=True)
    q_states = boundary_states(model, processor, layers, q_boundary, patch_span_length)
    qstar_states = boundary_states(model, processor, layers, qstar_boundary, patch_span_length)

    conditions = [
        ("q_recap_unpatched", q_boundary, None, None),
        ("qstar_recap_unpatched", qstar_boundary, None, None),
        ("q_recap_patched_with_qstar", q_boundary, args.patch_start_layer, qstar_states),
        ("qstar_recap_patched_with_q", qstar_boundary, args.patch_start_layer, q_states),
    ]
    results = {}
    for name, boundary, start_layer, source in conditions:
        print(f"Generating: {name}.", flush=True)
        results[name] = generate(
            model,
            processor,
            layers,
            boundary,
            args.max_new_tokens,
            patch_start_layer=start_layer,
            source_states=source,
            patch_span_length=patch_span_length if start_layer is not None else None,
        )

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "case_id": args.case,
        "manual_label": (label or {}).get("manual_label"),
        "patch_start_layer": args.patch_start_layer,
        "patch_scope": args.patch_scope,
        "patch_span_tokens": patch_span_length,
        "recap_span_tokens": recap_span_length,
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / (
        f"{args.case.replace('::', '__')}__generation_{args.patch_scope}_patch_"
        f"L{args.patch_start_layer}_{stamp}.json"
    )
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
