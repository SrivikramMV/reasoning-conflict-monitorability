from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classification import classify_answer_branch, classify_visible_branch
from common import (
    JsonlCheckpoint,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_ints,
    utc_now,
)
from runtime_utils import (
    build_candidate_texts,
    capture_hidden_positions,
    cleanup_model,
    completed_boundary_ids,
    continuation_logprob_ids,
    ensure_cuda_a100,
    generate_source_mask_variants,
    load_transformers_model,
)


CHECKPOINT_NAMES = {
    "candidate": "03_candidate_preference_standardized.jsonl",
    "hidden": "05_hidden_state_capture_standardized.jsonl",
    "masking": "06_source_masking_standardized.jsonl",
}


def stage_manifest(
    output_root: Path,
    model_key: str,
    stage: str,
    checkpoint: JsonlCheckpoint,
    expected: int,
    config_path: Path,
    profile: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "created_at_utc": utc_now(),
        "model_key": model_key,
        "stage": stage,
        "expected_records": expected,
        "observed_records": len(checkpoint.rows),
        "complete": len(checkpoint.rows) == expected,
        "checkpoint_file": str(checkpoint.path.relative_to(output_root)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint.path),
        "config_sha256": sha256_file(config_path),
        "model_id": profile["model_id"],
        "model_revision": profile["revision"],
    }
    if extra:
        payload.update(extra)
    path = output_root / "models" / model_key / "manifests" / f"{stage}_stage_manifest.json"
    atomic_write_json(path, payload)
    if not payload["complete"]:
        raise RuntimeError(f"{model_key} {stage}: {len(checkpoint.rows)} records, expected {expected}")


def run_candidate(
    bundle_root: Path,
    output_root: Path,
    model_key: str,
    model: Any,
    adapter: Any,
    config: dict[str, Any],
) -> None:
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    rows = [
        row
        for row in read_jsonl(
            bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl"
        )
        if row["condition"] == "full_cot_normal"
    ]
    subset = set(
        read_json(bundle_root / config["candidate_preference"]["sensitivity_subset_file"])[
            "pair_ids"
        ]
    )
    primary = config["candidate_preference"]["primary_template"]
    sensitivity = config["candidate_preference"]["sensitivity_templates"]
    checkpoint = JsonlCheckpoint(
        output_root
        / "models"
        / model_key
        / "checkpoints"
        / CHECKPOINT_NAMES["candidate"],
        "candidate_record_id",
    )
    preflight_path = (
        output_root / "models" / model_key / "manifests" / "candidate_preflight.json"
    )
    preflight_written = preflight_path.exists()
    if checkpoint.rows and not preflight_written:
        first = checkpoint.rows[0]
        values = first["q_score"]["token_logprobs"] + first["qstar_score"]["token_logprobs"]
        if not values or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("Stored candidate checkpoint fails resumed preflight")
        atomic_write_json(
            preflight_path,
            {
                "created_at_utc": utc_now(),
                "passed": True,
                "record_id": first["candidate_record_id"],
                "finite_token_scores": True,
                "resumed_from_checkpoint": True,
            },
        )
        preflight_written = True
    for parent in rows:
        boundary = completed_boundary_ids(parent, adapter)
        pair = pairs[parent["pair_id"]]
        templates = [primary] + (sensitivity if parent["pair_id"] in subset else [])
        for template in templates:
            record_id = f"{model_key}::{parent['pair_id']}::full_cot_normal::{template}"
            if not checkpoint.missing(record_id):
                continue
            preamble, q_text, qstar_text = build_candidate_texts(pair, template)
            prefix = boundary + adapter._encode(preamble)
            q_ids = adapter._encode(q_text)
            qstar_ids = adapter._encode(qstar_text)
            q_score = continuation_logprob_ids(model, prefix, q_ids)
            qstar_score = continuation_logprob_ids(model, prefix, qstar_ids)
            values = q_score["token_logprobs"] + qstar_score["token_logprobs"]
            if not values or not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"Non-finite candidate scores in {record_id}")
            record = {
                "candidate_record_id": record_id,
                "created_at_utc": utc_now(),
                "model_key": model_key,
                "pair_id": parent["pair_id"],
                "category": parent["category"],
                "condition": parent["condition"],
                "template": template,
                "template_id": template,
                "diagnostic_parent_id": parent["diagnostic_parent_id"],
                "boundary_token_count": len(boundary),
                "boundary_token_ids_sha256": sha256_ints(boundary),
                "preamble": preamble,
                "q_candidate": q_text,
                "qstar_candidate": qstar_text,
                "q_candidate_token_ids": q_ids,
                "qstar_candidate_token_ids": qstar_ids,
                "q_score": q_score,
                "qstar_score": qstar_score,
                "delta_total_qstar_minus_q": qstar_score["total_logprob"]
                - q_score["total_logprob"],
                "delta_average_qstar_minus_q": qstar_score["average_logprob"]
                - q_score["average_logprob"],
                "preferred_candidate": (
                    "QSTAR"
                    if qstar_score["average_logprob"] > q_score["average_logprob"]
                    else "Q"
                ),
            }
            checkpoint.append([record])
            if not preflight_written:
                atomic_write_json(
                    preflight_path,
                    {
                        "created_at_utc": utc_now(),
                        "passed": True,
                        "record_id": record_id,
                        "finite_token_scores": True,
                        "q_token_count": q_score["token_count"],
                        "qstar_token_count": qstar_score["token_count"],
                    },
                )
                preflight_written = True
    expected = len(rows) + len(subset) * len(sensitivity)
    stage_manifest(
        output_root,
        model_key,
        "candidate",
        checkpoint,
        expected,
        bundle_root / "config" / "standardized_diagnostics_config.json",
        config["models"][model_key],
    )


def run_hidden(
    bundle_root: Path,
    output_root: Path,
    model_key: str,
    model: Any,
    adapter: Any,
    config: dict[str, Any],
) -> None:
    rows = read_jsonl(
        bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl"
    )
    checkpoint = JsonlCheckpoint(
        output_root / "models" / model_key / "checkpoints" / CHECKPOINT_NAMES["hidden"],
        "hidden_record_id",
    )
    hidden_dir = output_root / "models" / model_key / "hidden_features"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    k_positions = [int(value) for value in config["hidden_state_capture"]["answer_prefix_positions"]]
    minimum = int(config["hidden_state_capture"]["minimum_answer_tokens"])
    preflight_path = (
        output_root / "models" / model_key / "manifests" / "hidden_preflight.json"
    )
    preflight_written = preflight_path.exists()
    if checkpoint.rows and not preflight_written:
        first = checkpoint.rows[0]
        npz_path = output_root / first["npz_file"]
        with np.load(npz_path) as archive:
            hidden = archive["hidden"]
            if not np.isfinite(hidden).all() or list(hidden.shape) != first["hidden_shape"]:
                raise RuntimeError("Stored hidden checkpoint fails resumed preflight")
        atomic_write_json(
            preflight_path,
            {
                "created_at_utc": utc_now(),
                "passed": True,
                "record_id": first["hidden_record_id"],
                "shape": first["hidden_shape"],
                "dtype": first["hidden_dtype"],
                "all_finite": True,
                "resumed_from_checkpoint": True,
            },
        )
        preflight_written = True
    for parent in rows:
        record_id = (
            f"{model_key}::{parent['pair_id']}::{parent['condition']}"
            "::answer_boundary_and_prefix"
        )
        if not checkpoint.missing(record_id):
            continue
        answer_ids = [int(value) for value in parent["final_answer_token_ids"]]
        if len(answer_ids) < minimum:
            raise RuntimeError(
                f"{record_id} has {len(answer_ids)} final-answer tokens; {minimum} required"
            )
        boundary = completed_boundary_ids(parent, adapter)
        prefix_answer = answer_ids[: max(k_positions)]
        full_ids = boundary + prefix_answer
        absolute_positions = [len(boundary) - 1 + value for value in k_positions]
        hidden, actual_positions = capture_hidden_positions(
            model, full_ids, absolute_positions
        )
        if not np.isfinite(hidden).all():
            raise RuntimeError(f"Non-finite hidden state in {record_id}")
        npz_name = record_id.replace("::", "__") + ".npz"
        npz_path = hidden_dir / npz_name
        np.savez_compressed(
            npz_path,
            hidden=hidden,
            absolute_positions=np.asarray(actual_positions, dtype=np.int64),
            answer_prefix_positions=np.asarray(k_positions, dtype=np.int64),
            boundary_token_count=np.asarray([len(boundary)], dtype=np.int64),
            input_token_ids=np.asarray(full_ids, dtype=np.int64),
        )
        record = {
            "hidden_record_id": record_id,
            "created_at_utc": utc_now(),
            "model_key": model_key,
            "pair_id": parent["pair_id"],
            "category": parent["category"],
            "condition": parent["condition"],
            "diagnostic_parent_id": parent["diagnostic_parent_id"],
            "boundary_token_count": len(boundary),
            "boundary_token_ids_sha256": sha256_ints(boundary),
            "captured_input_token_count": len(full_ids),
            "captured_input_token_ids_sha256": sha256_ints(full_ids),
            "answer_prefix_positions": k_positions,
            "absolute_positions": actual_positions,
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
            "all_finite": True,
            "representation_axis": "layer_input_residuals_plus_final_norm",
            "npz_file": str(npz_path.relative_to(output_root)).replace("\\", "/"),
            "npz_sha256": sha256_file(npz_path),
        }
        checkpoint.append([record])
        if not preflight_written:
            atomic_write_json(
                output_root
                / "models"
                / model_key
                / "manifests"
                / "hidden_preflight.json",
                {
                    "created_at_utc": utc_now(),
                    "passed": True,
                    "record_id": record_id,
                    "shape": list(hidden.shape),
                    "dtype": str(hidden.dtype),
                    "all_finite": True,
                    "positions": actual_positions,
                },
            )
            preflight_written = True
    stage_manifest(
        output_root,
        model_key,
        "hidden",
        checkpoint,
        len(rows),
        bundle_root / "config" / "standardized_diagnostics_config.json",
        config["models"][model_key],
        {"answer_prefix_positions": k_positions},
    )


def run_masking(
    bundle_root: Path,
    output_root: Path,
    model_key: str,
    model: Any,
    adapter: Any,
    config: dict[str, Any],
) -> None:
    pairs = {row["pair_id"]: row for row in read_jsonl(bundle_root / config["cohort_file"])}
    rows = read_jsonl(
        bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl"
    )
    checkpoint = JsonlCheckpoint(
        output_root / "models" / model_key / "checkpoints" / CHECKPOINT_NAMES["masking"],
        "mask_record_id",
    )
    settings = config["source_masking"]
    conditions = [str(value) for value in settings["conditions"]]
    max_new = int(settings["max_new_tokens"])
    endpoint = int(settings["first_endpoint_tokens"])
    preflight_path = (
        output_root / "models" / model_key / "manifests" / "masking_preflight.json"
    )
    preflight_written = preflight_path.exists()
    if checkpoint.rows and not preflight_written:
        first_parent = checkpoint.rows[0]["diagnostic_parent_id"]
        family = [
            row for row in checkpoint.rows if row["diagnostic_parent_id"] == first_parent
        ]
        if not family or not all(
            row["mask_provenance"].get("initial_prefix_mask_applied") for row in family
        ):
            raise RuntimeError("Stored masking checkpoint fails resumed preflight")
        atomic_write_json(
            preflight_path,
            {
                "created_at_utc": utc_now(),
                "passed": True,
                "diagnostic_parent_id": first_parent,
                "conditions_present": sorted(row["mask_condition"] for row in family),
                "initial_prefix_mask_applied_for_all": True,
                "resumed_from_checkpoint": True,
            },
        )
        preflight_written = True
    for parent in rows:
        missing = [
            condition
            for condition in conditions
            if checkpoint.missing(
                f"{model_key}::{parent['pair_id']}::{parent['condition']}::mask_{condition}"
            )
        ]
        if not missing:
            continue
                                                                               
                                                                     
        boundary = completed_boundary_ids(parent, adapter)
        started = time.time()
        variants = generate_source_mask_variants(
            model,
            adapter,
            parent,
            boundary,
            conditions,
            max_new,
            endpoint,
        )
        elapsed = time.time() - started
        pair = pairs[parent["pair_id"]]
        parent_branch = classify_answer_branch(
            parent.get("final_answer", ""),
            pair["original_answer_spec"],
            pair["counterfactual_answer_spec"],
        )
        records = []
        for condition in missing:
            record_id = (
                f"{model_key}::{parent['pair_id']}::{parent['condition']}"
                f"::mask_{condition}"
            )
            parsed = variants[condition]
            final_branch = classify_answer_branch(
                parsed["final_answer"],
                pair["original_answer_spec"],
                pair["counterfactual_answer_spec"],
            )
            parent_label = parent_branch["branch"]
            observed_label = final_branch["branch"]
            classifiable = parent_label in {"Q", "QSTAR"}
            record = {
                **parsed,
                "mask_record_id": record_id,
                "created_at_utc": utc_now(),
                "model_key": model_key,
                "pair_id": parent["pair_id"],
                "category": parent["category"],
                "condition": parent["condition"],
                "mask_condition": condition,
                "diagnostic_parent_id": parent["diagnostic_parent_id"],
                "boundary_token_count": len(boundary),
                "boundary_token_ids_sha256": sha256_ints(boundary),
                "elapsed_seconds_for_four_variant_batch": elapsed,
                "automatic_visible_branch": classify_visible_branch(
                    parsed["final_answer"], pair
                ),
                "automatic_final_branch": final_branch,
                "parent_automatic_final_branch": parent_branch,
                "unmasked_control_classifiable": classifiable if condition == "none" else None,
                "unmasked_control_branch_match": (
                    observed_label == parent_label
                    if condition == "none" and classifiable
                    else None
                ),
            }
            records.append(record)
        checkpoint.append(records)

        if not preflight_written:
            all_initial = all(
                variants[value]["mask_provenance"]["initial_prefix_mask_applied"]
                for value in conditions
            )
            none_zeros = variants["none"]["mask_provenance"]["masked_position_count"]
            if not all_initial or none_zeros != 0:
                raise RuntimeError("Source-mask implementation preflight failed")
            atomic_write_json(
                output_root
                / "models"
                / model_key
                / "manifests"
                / "masking_preflight.json",
                {
                    "created_at_utc": utc_now(),
                    "passed": True,
                    "diagnostic_parent_id": parent["diagnostic_parent_id"],
                    "conditions": conditions,
                    "initial_prefix_mask_applied_for_all": all_initial,
                    "none_masked_position_count": none_zeros,
                    "question_masked_position_count": variants["question"][
                        "mask_provenance"
                    ]["masked_position_count"],
                    "trace_masked_position_count": variants["trace"][
                        "mask_provenance"
                    ]["masked_position_count"],
                    "joint_masked_position_count": variants["joint"][
                        "mask_provenance"
                    ]["masked_position_count"],
                },
            )
            preflight_written = True
    stage_manifest(
        output_root,
        model_key,
        "masking",
        checkpoint,
        len(rows) * len(conditions),
        bundle_root / "config" / "standardized_diagnostics_config.json",
        config["models"][model_key],
        {
            "mask_applied_on_initial_prefix_pass": True,
            "variants_batched_per_shared_prefix": len(conditions),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--stages", nargs="+", default=["candidate", "hidden", "masking"]
    )
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(
        bundle_root / "config" / "standardized_diagnostics_config.json"
    )
    profile = config["models"][args.model_key]
    gpu = ensure_cuda_a100(config)
    processor_or_tokenizer, model, adapter = load_transformers_model(profile)
    validation = adapter.validate_template()
    manifest_dir = output_root / "models" / args.model_key / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    text_config = getattr(model.config, "text_config", model.config)
    atomic_write_json(
        manifest_dir / "transformers_runtime.json",
        {
            "created_at_utc": utc_now(),
            "model_key": args.model_key,
            "model_id": profile["model_id"],
            "model_revision": profile["revision"],
            "model_profile": profile,
            "gpu": gpu,
            "model_class": model.__class__.__name__,
            "tokenizer_class": adapter.tokenizer.__class__.__name__,
            "num_hidden_layers": int(getattr(text_config, "num_hidden_layers", -1)),
            "hidden_size": int(getattr(text_config, "hidden_size", -1)),
            "adapter_validation": validation,
            "requested_stages": args.stages,
        },
    )
    try:
        if "candidate" in args.stages:
            run_candidate(
                bundle_root, output_root, args.model_key, model, adapter, config
            )
        if "hidden" in args.stages:
            run_hidden(
                bundle_root, output_root, args.model_key, model, adapter, config
            )
        if "masking" in args.stages:
            run_masking(
                bundle_root, output_root, args.model_key, model, adapter, config
            )
    finally:
        cleanup_model(model)


if __name__ == "__main__":
    main()
