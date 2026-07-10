







from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_boundary_state_batch import changed_line_opening
from local_gemma_follow_bypass_probe import (
    MODEL_ID,
    OUT_DIR,
    build_gemma_prompt,
    ensure_answer_channel,
    load_case,
    strip_answer_channel,
)


DEFAULT_CASE = "nat_lin3_004::full_cot_normal"
DEFAULT_LAYERS = [0, 10, 20, 25, 30, 34]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def neutral_system_recap(packet: dict[str, Any], use_qstar: bool) -> str:
    question = packet["counterfactual_question"] if use_qstar else packet["original_question"]
    equation_lines = [line.strip() for line in question.splitlines() if "=" in line]
    recap = "\n".join(f"    ({i}) ${line}$" for i, line in enumerate(equation_lines, start=1))
    thought = strip_answer_channel(packet["injected_cot_full"])
    return ensure_answer_channel(
        thought.rstrip()
        + "\n\nFinal consistency recap of the active system:\n"
        + recap
        + "\nThe final response should present the solution for this active system."
    )


def text_layers(model: Any) -> Any:
    base = model.model
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        return base.language_model.layers
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError("Could not locate Gemma text decoder layers.")


def boundary_states(
    model: Any,
    processor: Any,
    layers: Any,
    boundary_text: str,
    span_length: int,
) -> list[torch.Tensor]:

    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    for layer_index, layer in enumerate(layers):
        def capture_input(module: Any, args: tuple[Any, ...], index: int = layer_index):
            captured[index] = args[0][0, -span_length:, :].detach().float().cpu()

        handles.append(layer.register_forward_pre_hook(capture_input))

    inputs = processor(text=boundary_text, return_tensors="pt").to(model.device)
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if any(state is None for state in captured):
        raise RuntimeError("Failed to capture one or more decoder-layer inputs.")
    states = [state for state in captured if state is not None]
    del out
    del inputs
    torch.cuda.empty_cache()
    return states


def candidate_logprob(
    model: Any,
    processor: Any,
    layers: Any,
    boundary_text: str,
    continuation_prefix: str,
    continuation_target: str,
    patch_start_layer: int | None = None,
    source_states: list[torch.Tensor] | None = None,
    patch_span_length: int | None = None,
) -> dict[str, Any]:
    boundary_inputs = processor(text=boundary_text, return_tensors="pt")
    prefix_text = boundary_text + continuation_prefix
    prefix_inputs = processor(text=prefix_text, return_tensors="pt")
    full_inputs = processor(text=prefix_text + continuation_target, return_tensors="pt").to(model.device)

    boundary_length = int(boundary_inputs["input_ids"].shape[-1])
    prefix_len = int(prefix_inputs["input_ids"].shape[-1])
    target_ids = full_inputs["input_ids"][:, prefix_len:]
    if target_ids.numel() == 0:
        raise ValueError("Candidate continuation produced no target tokens.")

    handles = []
    if patch_start_layer is not None:
        if source_states is None or patch_span_length is None:
            raise ValueError("source_states and patch_span_length are required for patching.")
        span_start = boundary_length - patch_span_length
        span_end = boundary_length

        for layer_index in range(patch_start_layer, len(layers)):
            patch = source_states[layer_index].to(device=model.device, dtype=model.dtype)

            def replace_recap_span(
                module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                value: torch.Tensor = patch,
            ):
                hidden = args[0]
                replaced = hidden.clone()
                replaced[:, span_start:span_end, :] = value
                return (replaced, *args[1:]), kwargs

            handles.append(
                layers[layer_index].register_forward_pre_hook(replace_recap_span, with_kwargs=True)
            )

    try:
        with torch.inference_mode():
            out = model(**full_inputs, use_cache=False)
            logits = out.logits[:, prefix_len - 1 : -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    finally:
        for handle in handles:
            handle.remove()

    values = token_log_probs[0].detach().cpu().tolist()
    total = float(sum(values))
    result = {
        "text": continuation_target,
        "token_count": len(values),
        "total_logprob": total,
        "avg_logprob": total / len(values),
        "perplexity": math.exp(-total / len(values)),
        "token_logprobs": values,
    }
    del out
    del full_inputs
    torch.cuda.empty_cache()
    return result


def score_condition(
    model: Any,
    processor: Any,
    layers: Any,
    packet: dict[str, Any],
    boundary_text: str,
    patch_start_layer: int | None = None,
    source_states: list[torch.Tensor] | None = None,
    patch_span_length: int | None = None,
) -> dict[str, Any]:
    changed_line_number = int((packet.get("changed_line") or {}).get("line_number") or 2)
    q_preamble, q_target = changed_line_opening(packet["original_question"], changed_line_number)
    qstar_preamble, qstar_target = changed_line_opening(packet["counterfactual_question"], changed_line_number)

    q = candidate_logprob(
        model,
        processor,
        layers,
        boundary_text,
        q_preamble,
        q_target,
        patch_start_layer,
        source_states,
        patch_span_length,
    )
    qstar = candidate_logprob(
        model,
        processor,
        layers,
        boundary_text,
        qstar_preamble,
        qstar_target,
        patch_start_layer,
        source_states,
        patch_span_length,
    )
    return {
        "q": q,
        "qstar": qstar,
        "delta_avg_qstar_minus_q": qstar["avg_logprob"] - q["avg_logprob"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "activation_patching")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

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
    for layer in args.layers:
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"Layer {layer} is outside 0..{len(layers) - 1}.")

    prompt = build_gemma_prompt(processor, packet["original_question"])
    thought_only = strip_answer_channel(packet["injected_cot_full"]).rstrip()
    unrecapped_boundary = prompt + thought_only
    q_boundary = prompt + neutral_system_recap(packet, use_qstar=False)
    qstar_boundary = prompt + neutral_system_recap(packet, use_qstar=True)

    unrecapped_length = int(processor(text=unrecapped_boundary, return_tensors="pt")["input_ids"].shape[-1])
    q_boundary_length = int(processor(text=q_boundary, return_tensors="pt")["input_ids"].shape[-1])
    qstar_boundary_length = int(processor(text=qstar_boundary, return_tensors="pt")["input_ids"].shape[-1])
    if q_boundary_length != qstar_boundary_length:
        raise ValueError(
            f"Matched recap token lengths differ: Q={q_boundary_length}, Q*={qstar_boundary_length}."
        )
    patch_span_length = q_boundary_length - unrecapped_length
    if patch_span_length <= 0:
        raise ValueError("Could not identify a non-empty recap span.")

    print(f"Loaded {MODEL_ID}; extracting boundary states for {args.case}.", flush=True)
    q_states = boundary_states(model, processor, layers, q_boundary, patch_span_length)
    qstar_states = boundary_states(model, processor, layers, qstar_boundary, patch_span_length)

    print("Scoring unpatched matched conditions.", flush=True)
    baseline_q = score_condition(model, processor, layers, packet, q_boundary)
    baseline_qstar = score_condition(model, processor, layers, packet, qstar_boundary)

    output: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "model_id": MODEL_ID,
        "case_id": args.case,
        "manual_label": (label or {}).get("manual_label"),
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "changed_line": packet.get("changed_line"),
        "layers_tested": args.layers,
        "patch_type": "aligned_recap_span_at_every_layer_from_start_layer_through_final_layer",
        "patch_span_tokens": patch_span_length,
        "conditions": {
            "q_recap_tail": q_boundary[-1200:],
            "qstar_recap_tail": qstar_boundary[-1200:],
        },
        "baselines": {
            "q_recap": baseline_q,
            "qstar_recap": baseline_qstar,
        },
        "patches": [],
    }

    base_q_delta = baseline_q["delta_avg_qstar_minus_q"]
    base_qstar_delta = baseline_qstar["delta_avg_qstar_minus_q"]
    denominator = base_qstar_delta - base_q_delta

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.case.replace('::', '__')}__{stamp}.json"

    for layer in args.layers:
        print(f"Layer {layer}: Q -> Q* and Q* -> Q boundary patches.", flush=True)
        q_to_qstar = score_condition(
            model,
            processor,
            layers,
            packet,
            q_boundary,
            patch_start_layer=layer,
            source_states=qstar_states,
            patch_span_length=patch_span_length,
        )
        qstar_to_q = score_condition(
            model,
            processor,
            layers,
            packet,
            qstar_boundary,
            patch_start_layer=layer,
            source_states=q_states,
            patch_span_length=patch_span_length,
        )
        row = {
            "layer": layer,
            "q_recap_patched_with_qstar_state": q_to_qstar,
            "qstar_recap_patched_with_q_state": qstar_to_q,
            "forward_shift_toward_qstar": q_to_qstar["delta_avg_qstar_minus_q"] - base_q_delta,
            "reverse_shift_toward_q": base_qstar_delta - qstar_to_q["delta_avg_qstar_minus_q"],
            "forward_fraction_of_baseline_gap": (
                (q_to_qstar["delta_avg_qstar_minus_q"] - base_q_delta) / denominator
                if abs(denominator) > 1e-9
                else None
            ),
            "reverse_fraction_of_baseline_gap": (
                (base_qstar_delta - qstar_to_q["delta_avg_qstar_minus_q"]) / denominator
                if abs(denominator) > 1e-9
                else None
            ),
            "source_state_cosine": float(
                torch.nn.functional.cosine_similarity(
                    q_states[layer].flatten(), qstar_states[layer].flatten(), dim=0
                ).item()
            ),
            "q_state_norm": float(q_states[layer].norm().item()),
            "qstar_state_norm": float(qstar_states[layer].norm().item()),
        }
        output["patches"].append(row)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output": str(out_path),
        "baseline_q_delta": base_q_delta,
        "baseline_qstar_delta": base_qstar_delta,
        "patch_summary": [
            {
                "layer": row["layer"],
                "forward_delta": row["q_recap_patched_with_qstar_state"]["delta_avg_qstar_minus_q"],
                "reverse_delta": row["qstar_recap_patched_with_q_state"]["delta_avg_qstar_minus_q"],
                "forward_fraction": row["forward_fraction_of_baseline_gap"],
                "reverse_fraction": row["reverse_fraction_of_baseline_gap"],
            }
            for row in output["patches"]
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
