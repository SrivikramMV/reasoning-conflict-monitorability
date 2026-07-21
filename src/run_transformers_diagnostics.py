from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classification import classify_answer_branch, classify_visible_branch
from common import JsonlCheckpoint, atomic_write_json, read_json, read_jsonl, sha256_file, sha256_ints, stable_seed, utc_now
from runtime_utils import (
    build_candidate_texts,
    capture_hidden_positions,
    cleanup_model,
    completed_boundary_ids,
    continuation_logprob_ids,
    decode,
    ensure_cuda_a100,
    generate_with_source_mask,
    load_transformers_model,
)


def run_candidate(bundle_root: Path, output_root: Path, model_key: str, model: Any, adapter: Any, config: dict[str, Any]) -> None:
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    parent_rows = read_jsonl(bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl")
    full_rows = [row for row in parent_rows if row["condition"] == "full_cot_normal"]
    subset = set(read_json(bundle_root / config["candidate_preference"]["sensitivity_subset_file"])["pair_ids"])
    primary_template = config["candidate_preference"]["primary_template"]
    templates_for = {
        row["pair_id"]: [primary_template] + (config["candidate_preference"]["sensitivity_templates"] if row["pair_id"] in subset else [])
        for row in full_rows
    }
    checkpoint = JsonlCheckpoint(output_root / "models" / model_key / "checkpoints" / "03_candidate_preference_standardized.jsonl", "candidate_record_id")
    for row in full_rows:
        boundary = completed_boundary_ids(row, adapter)
        pair = pairs[row["pair_id"]]
        for template in templates_for[row["pair_id"]]:
            record_id = f"{model_key}::{row['pair_id']}::full_cot_normal::{template}"
            if not checkpoint.missing(record_id):
                continue
            preamble, q_text, qstar_text = build_candidate_texts(pair, template)
            preamble_ids = adapter._encode(preamble)
            q_ids = adapter._encode(q_text)
            qstar_ids = adapter._encode(qstar_text)
            prefix = boundary + preamble_ids
            q_score = continuation_logprob_ids(model, prefix, q_ids)
            qstar_score = continuation_logprob_ids(model, prefix, qstar_ids)
            row_out = {
                "candidate_record_id": record_id,
                "created_at_utc": utc_now(),
                "model_key": model_key,
                "pair_id": row["pair_id"],
                "category": row["category"],
                "condition": row["condition"],
                "template": template,
                "diagnostic_parent_id": row["diagnostic_parent_id"],
                "boundary_token_count": len(boundary),
                "preamble": preamble,
                "q_candidate": q_text,
                "qstar_candidate": qstar_text,
                "q_score": q_score,
                "qstar_score": qstar_score,
                "delta_average_qstar_minus_q": qstar_score["average_logprob"] - q_score["average_logprob"],
                "preferred_candidate": "QSTAR" if qstar_score["average_logprob"] > q_score["average_logprob"] else "Q",
            }
            checkpoint.append([row_out])


def sample_with_transformers(model: Any, adapter: Any, boundary: list[int], max_new_tokens: int, temperature: float, top_p: float, count: int, seed: int) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    input_tensor = torch.tensor([boundary], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    kwargs = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_return_sequences": int(count),
        "eos_token_id": adapter.markers.terminal_ids,
        "pad_token_id": adapter.markers.pad_token_id,
        "use_cache": True,
    }
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        with torch.inference_mode():
            output = model.generate(**kwargs)
    values = []
    for sequence in output:
        generated = [int(v) for v in sequence[len(boundary):].tolist()]
        values.append({"generated_token_ids": generated})
    del output, input_tensor, attention_mask
    return values


def run_sampling_gemma(bundle_root: Path, output_root: Path, model_key: str, model: Any, adapter: Any, config: dict[str, Any]) -> None:
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    parent_rows = [row for row in read_jsonl(bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl") if row["condition"] == "full_cot_normal"]
    checkpoint = JsonlCheckpoint(output_root / "models" / model_key / "checkpoints" / "04_boundary_sampling_standardized.jsonl", "sample_record_id")
    settings = config["boundary_sampling"]
    for row in parent_rows:
        boundary = completed_boundary_ids(row, adapter)
        for temp in settings["temperatures"]:
            for sample_index in range(settings["samples_per_temperature"]):
                record_id = f"{model_key}::{row['pair_id']}::full_cot_normal::temp_{temp}::sample_{sample_index:02d}"
                if not checkpoint.missing(record_id):
                    continue
                seed = stable_seed(config["models"][model_key]["revision"], record_id, int(config["seeds"]["sampling"]))
                generated = sample_with_transformers(model, adapter, boundary, settings["max_new_tokens"], temp, settings["top_p"], 1, seed)[0]["generated_token_ids"]
                parsed = parse_sample(adapter, row, pairs[row["pair_id"]], generated, settings["max_new_tokens"])
                parsed.update({
                    "sample_record_id": record_id,
                    "created_at_utc": utc_now(),
                    "model_key": model_key,
                    "pair_id": row["pair_id"],
                    "category": row["category"],
                    "condition": row["condition"],
                    "temperature": temp,
                    "sample_index": sample_index,
                    "seed": seed,
                    "diagnostic_parent_id": row["diagnostic_parent_id"],
                })
                checkpoint.append([parsed])


def parse_sample(adapter: Any, row: dict[str, Any], pair: dict[str, Any], generated: list[int], max_new_tokens: int) -> dict[str, Any]:
    answer_ids = []
    stop_reason = "max_new_tokens" if len(generated) >= max_new_tokens else "generation_stopped_without_registered_token"
    for value in generated:
        if value in set(adapter.markers.terminal_ids):
            stop_reason = "terminal"
            break
        answer_ids.append(int(value))
    text = decode(adapter, answer_ids)
    branch = classify_visible_branch(text, pair)
    return {
        "generated_token_ids": [int(v) for v in generated],
        "generated_token_count": len(generated),
        "final_answer_token_ids": answer_ids,
        "final_answer": text.strip(),
        "truncated": stop_reason == "max_new_tokens",
        "stop_reason": stop_reason,
        "automatic_visible_branch": branch,
        "automatic_final_branch": classify_answer_branch(text, pair["original_answer_spec"], pair["counterfactual_answer_spec"]),
    }


def run_hidden(bundle_root: Path, output_root: Path, model_key: str, model: Any, adapter: Any, config: dict[str, Any]) -> None:
    parent_rows = read_jsonl(bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl")
    checkpoint = JsonlCheckpoint(output_root / "models" / model_key / "checkpoints" / "05_hidden_state_capture_standardized.jsonl", "hidden_record_id")
    hidden_dir = output_root / "models" / model_key / "hidden_features"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    positions_config = config["hidden_state_capture"]["answer_prefix_positions"]
    for row in parent_rows:
        record_id = f"{model_key}::{row['pair_id']}::{row['condition']}::answer_boundary_and_prefix"
        if not checkpoint.missing(record_id):
            continue
        boundary = completed_boundary_ids(row, adapter)
        prefix_answer = [int(v) for v in row["final_answer_token_ids"][:max(positions_config)]]
        full = boundary + prefix_answer
        positions = [len(boundary) - 1 + int(k) for k in positions_config]
        hidden, actual_positions = capture_hidden_positions(model, full, positions)
        npz_name = record_id.replace("::", "__") + ".npz"
        npz_path = hidden_dir / npz_name
        np.savez_compressed(npz_path, hidden=hidden, positions=np.array(actual_positions), k=np.array(positions_config), boundary_token_count=np.array([len(boundary)]))
        checkpoint.append([{
            "hidden_record_id": record_id,
            "created_at_utc": utc_now(),
            "model_key": model_key,
            "pair_id": row["pair_id"],
            "category": row["category"],
            "condition": row["condition"],
            "diagnostic_parent_id": row["diagnostic_parent_id"],
            "boundary_token_count": len(boundary),
            "answer_prefix_positions": positions_config,
            "absolute_positions": actual_positions,
            "hidden_shape": list(hidden.shape),
            "npz_file": str(npz_path.relative_to(output_root)).replace("\\\\", "/"),
            "npz_sha256": sha256_file(npz_path),
        }])


def run_masking(bundle_root: Path, output_root: Path, model_key: str, model: Any, adapter: Any, config: dict[str, Any]) -> None:
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    parent_rows = read_jsonl(bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl")
    checkpoint = JsonlCheckpoint(output_root / "models" / model_key / "checkpoints" / "06_source_masking_standardized.jsonl", "mask_record_id")
    max_new = int(config["source_masking"]["max_new_tokens"])
    for row in parent_rows:
        boundary = completed_boundary_ids(row, adapter)
        for mask_condition in config["source_masking"]["conditions"]:
            record_id = f"{model_key}::{row['pair_id']}::{row['condition']}::mask_{mask_condition}"
            if not checkpoint.missing(record_id):
                continue
            torch.manual_seed(stable_seed(config["models"][model_key]["revision"], record_id, int(config["seeds"]["masking"])))
            started = time.time()
            parsed = generate_with_source_mask(model, adapter, row, boundary, mask_condition, max_new)
            elapsed = time.time() - started
            pair = pairs[row["pair_id"]]
            parsed.update({
                "mask_record_id": record_id,
                "created_at_utc": utc_now(),
                "model_key": model_key,
                "pair_id": row["pair_id"],
                "category": row["category"],
                "condition": row["condition"],
                "mask_condition": mask_condition,
                "diagnostic_parent_id": row["diagnostic_parent_id"],
                "elapsed_seconds": elapsed,
                "automatic_visible_branch": classify_visible_branch(parsed["final_answer"], pair),
                "automatic_final_branch": classify_answer_branch(parsed["final_answer"], pair["original_answer_spec"], pair["counterfactual_answer_spec"]),
            })
            checkpoint.append([parsed])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--stages", nargs="+", default=["candidate", "hidden", "masking"])
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(bundle_root / "config" / "standardized_diagnostics_config.json")
    profile = config["models"][args.model_key]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "models" / args.model_key / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    gpu = ensure_cuda_a100(config)
    processor_or_tokenizer, model, adapter = load_transformers_model(profile)
    atomic_write_json(manifest_dir / "transformers_runtime.json", {
        "created_at_utc": utc_now(),
        "model_key": args.model_key,
        "model_profile": profile,
        "gpu": gpu,
        "model_class": model.__class__.__name__,
        "tokenizer_class": adapter.tokenizer.__class__.__name__,
        "num_hidden_layers": int(getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", -1)),
        "hidden_size": int(getattr(getattr(model.config, "text_config", model.config), "hidden_size", -1)),
    })
    try:
        if "candidate" in args.stages:
            run_candidate(bundle_root, output_root, args.model_key, model, adapter, config)
        if "sampling" in args.stages:
            run_sampling_gemma(bundle_root, output_root, args.model_key, model, adapter, config)
        if "hidden" in args.stages:
            run_hidden(bundle_root, output_root, args.model_key, model, adapter, config)
        if "masking" in args.stages:
            run_masking(bundle_root, output_root, args.model_key, model, adapter, config)
    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
