

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from local_gemma_activation_patch_pilot import text_layers
from local_gemma_candidate_template_smoke import candidate_pair, classify_generated
from local_gemma_follow_bypass_probe import MODEL_ID, ROOT, build_gemma_prompt
from local_gemma_heldout_boundary_extract import cohort, completed_cot


DEFAULT_K = [0, 1, 2, 4, 8, 16, 32, 64]
CAPTURE_LAYER_INPUTS = {20: "after_block_19", 33: "after_block_32"}
EXTRACTION_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return value.replace("::", "__").replace("/", "_")


def first_visible_state(
    tokenizer: Any,
    answer_ids: torch.Tensor,
    q_target: str,
    qstar_target: str,
) -> tuple[int | None, str]:
    for count in range(1, int(answer_ids.numel()) + 1):
        prefix = tokenizer.decode(
            answer_ids[:count], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        state = classify_generated(prefix, q_target, qstar_target)
        if state in {"Q", "QSTAR"}:
            return count, state
    return None, "OTHER"


def capture(
    model: Any,
    processor: Any,
    layers: Any,
    full_inputs: dict[str, torch.Tensor],
    positions: list[int],
) -> dict[str, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer_index, name in CAPTURE_LAYER_INPUTS.items():
        def hook(module: Any, args: tuple[Any, ...], key: str = name) -> None:
            captured[key] = args[0][0, positions, :].detach().float().cpu()

        handles.append(layers[layer_index].register_forward_pre_hook(hook))

    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model

    def norm_hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["final_norm"] = output[0, positions, :].detach().float().cpu()

    handles.append(text_model.norm.register_forward_hook(norm_hook))
    try:
        with torch.inference_mode():
            output = model(**full_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = set(CAPTURE_LAYER_INPUTS.values()) | {"final_norm"}
    if set(captured) != expected:
        raise RuntimeError(f"Missing captures: {sorted(expected - set(captured))}")
    arrays = {name: value.numpy() for name, value in captured.items()}
    del output
    torch.cuda.empty_cache()
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Results" / "dissertation_smoke_tests" / "visibility_latency" / "features",
    )
    args = parser.parse_args()
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
    layers = text_layers(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    rows = cohort()
    for row_index, (packet, label) in enumerate(rows, start=1):
        case_id = packet["intervention_id"]
        stem = safe_name(case_id)
        json_path = args.out_dir / f"{stem}.json"
        npz_path = args.out_dir / f"{stem}.npz"
        print(f"[{row_index}/{len(rows)}] {case_id}", flush=True)

        boundary = build_gemma_prompt(processor, packet["original_question"]) + completed_cot(packet)
        answer = packet["final_answer_full"]
        boundary_inputs = processor(text=boundary, return_tensors="pt")
        full_inputs_cpu = processor(text=boundary + answer, return_tensors="pt")
        boundary_ids = boundary_inputs["input_ids"][0]
        full_ids = full_inputs_cpu["input_ids"][0]
        boundary_len = int(boundary_ids.numel())
        if not torch.equal(full_ids[:boundary_len], boundary_ids):
            raise RuntimeError(f"Boundary tokenisation changed after appending answer for {case_id}")
        answer_ids = full_ids[boundary_len:]
        valid_k = [value for value in sorted(set(args.k)) if value <= int(answer_ids.numel())]
        positions = [boundary_len - 1 + value for value in valid_k]

        preamble, q_target, qstar_target = candidate_pair(packet, "standard_system")
        reveal_token, reveal_state = first_visible_state(
            processor.tokenizer, answer_ids, q_target, qstar_target
        )
        prefix_text = {
            str(value): processor.tokenizer.decode(
                answer_ids[:value],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for value in valid_k
        }

        full_inputs = {key: value.to(model.device) for key, value in full_inputs_cpu.items()}
        arrays = capture(model, processor, layers, full_inputs, positions)
        np.savez_compressed(npz_path, k=np.asarray(valid_k, dtype=np.int32), **arrays)

        metadata = {
            "created_at_utc": utc_now(),
            "extraction_version": EXTRACTION_VERSION,
            "model_id": MODEL_ID,
            "case_id": case_id,
            "item_id": case_id.split("::", 1)[0],
            "variant": case_id.split("::", 1)[1],
            "manual_label": label["manual_label"],
            "label": 1 if label["manual_label"] == "FOLLOW" else 0,
            "boundary_token_count": boundary_len,
            "answer_token_count": int(answer_ids.numel()),
            "k": valid_k,
            "answer_prefix_by_k": prefix_text,
            "first_exact_state_token": reveal_token,
            "first_exact_state": reveal_state,
            "q_target": q_target,
            "qstar_target": qstar_target,
            "visible_base_text": packet["original_question"] + "\n\n" + completed_cot(packet),
        }
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(str(json_path))
        del full_inputs, full_inputs_cpu, boundary_inputs
        torch.cuda.empty_cache()

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
