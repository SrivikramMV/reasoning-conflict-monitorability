

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_activation_patch_pilot import text_layers
from local_gemma_boundary_hidden_separability import boundary_delta_score
from local_gemma_follow_bypass_probe import (
    LABELS_JSON,
    MODEL_ID,
    OUT_DIR,
    PACKETS_JSON,
    build_gemma_prompt,
    ensure_answer_channel,
    load_json,
    strip_answer_channel,
)


ELIGIBLE_VARIANTS = {"cot_50_percent", "cot_90_percent", "full_cot_normal"}
LABEL_MAP = {"FOLLOW": 1, "SILENT_BYPASS_REANCHOR": 0}
EXTRACTION_VERSION = 2


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def cohort() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    packets = {row["intervention_id"]: row for row in load_json(PACKETS_JSON)}
    rows = []
    for label in load_json(LABELS_JSON):
        intervention_id = label["intervention_id"]
        item_id, variant = intervention_id.split("::", 1)
        if not item_id.startswith("nat_lin3_"):
            continue
        if variant not in ELIGIBLE_VARIANTS or label.get("manual_label") not in LABEL_MAP:
            continue
        rows.append((packets[intervention_id], label))
    return sorted(rows, key=lambda pair: pair[0]["intervention_id"])


def capture_boundary(
    model: Any,
    processor: Any,
    layers: Any,
    boundary_text: str,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    captured: list[torch.Tensor | None] = [None] * len(layers)
    final_norm: torch.Tensor | None = None
    handles = []

    for layer_index, layer in enumerate(layers):
        def capture_input(module: Any, args: tuple[Any, ...], index: int = layer_index):
            captured[index] = args[0][0, -1, :].detach().float().cpu()

        handles.append(layer.register_forward_pre_hook(capture_input))

    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model

    def capture_norm(module: Any, args: tuple[Any, ...], output: torch.Tensor):
        nonlocal final_norm
        final_norm = output[0, -1, :].detach().float().cpu()

    handles.append(text_model.norm.register_forward_hook(capture_norm))
    inputs = processor(text=boundary_text, return_tensors="pt").to(model.device)
    token_count = int(inputs["input_ids"].shape[-1])
    last_token_id = int(inputs["input_ids"][0, -1])
    last_token = processor.tokenizer.decode(
        [last_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if any(value is None for value in captured) or final_norm is None:
        raise RuntimeError("Failed to capture all boundary activations.")
    layer_inputs = np.stack([value.numpy() for value in captured if value is not None], axis=0)
    final = final_norm.numpy()
    del out
    del inputs
    torch.cuda.empty_cache()
    return layer_inputs, final, token_count, last_token


def completed_cot(packet: dict[str, Any]) -> str:
    injected_thought = strip_answer_channel(packet["injected_cot_full"])
    generated = packet.get("generated_cot_continuation_full") or ""
    return ensure_answer_channel(injected_thought + generated)


def visible_features(packet: dict[str, Any], cot: str, token_count: int) -> dict[str, Any]:
    variant = packet["intervention_id"].split("::", 1)[1]
    depth = {"cot_50_percent": 0.5, "cot_90_percent": 0.9, "full_cot_normal": 1.0}[variant]
    tail = cot[-1200:]
    return {
        "injection_depth": depth,
        "boundary_token_count": token_count,
        "cot_chars": len(cot),
        "cot_lines": len(cot.splitlines()),
        "cot_fraction_slashes": cot.count("/"),
        "cot_verify_mentions": len(re.findall(r"verif|check", cot, flags=re.IGNORECASE)),
        "cot_question_mentions": len(re.findall(r"question|prompt|problem", cot, flags=re.IGNORECASE)),
        "cot_wait_mentions": len(re.findall(r"\bwait\b|re-?read|double.?check", cot, flags=re.IGNORECASE)),
        "visible_text": packet["original_question"] + "\n\n" + cot,
        "completed_cot_full": cot,
        "cot_tail": tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR / "heldout_boundary_prediction" / "features",
    )
    parser.add_argument("--skip-delta", action="store_true")
    args = parser.parse_args()

    rows = cohort()
    print(f"Eligible cohort: {len(rows)} rows.", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    layers = text_layers(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, (packet, label) in enumerate(rows, start=1):
        intervention_id = packet["intervention_id"]
        item_id, variant = intervention_id.split("::", 1)
        stem = safe_name(intervention_id)
        npz_path = args.out_dir / f"{stem}.npz"
        json_path = args.out_dir / f"{stem}.json"
        if npz_path.exists() and json_path.exists():
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            if existing.get("extraction_version") == EXTRACTION_VERSION:
                print(f"[{index}/{len(rows)}] cached {intervention_id}", flush=True)
                manifest.append(str(json_path))
                continue

        print(f"[{index}/{len(rows)}] extracting {intervention_id}", flush=True)
        cot = completed_cot(packet)
        boundary_text = build_gemma_prompt(processor, packet["original_question"]) + cot
        layer_inputs, final_norm, token_count, last_token = capture_boundary(
            model, processor, layers, boundary_text
        )
        delta = None if args.skip_delta else boundary_delta_score(model, processor, packet, boundary_text)
        features = visible_features(packet, cot, token_count)
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "extraction_version": EXTRACTION_VERSION,
            "model_id": MODEL_ID,
            "intervention_id": intervention_id,
            "item_id": item_id,
            "variant": variant,
            "manual_label": label["manual_label"],
            "label": LABEL_MAP[label["manual_label"]],
            "manual_subtype": label.get("manual_subtype"),
            "original_question": packet["original_question"],
            "counterfactual_question": packet["counterfactual_question"],
            "original_answer": packet["original_answer"],
            "counterfactual_answer": packet["counterfactual_answer"],
            "changed_line": packet.get("changed_line"),
            "last_prefill_token": last_token,
            "boundary_delta_qstar_minus_q": delta,
            **features,
        }
        np.savez_compressed(npz_path, layer_inputs=layer_inputs, final_norm=final_norm)
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(str(json_path))

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
