from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    stable_seed,
    utc_now,
)


CHECKPOINTS = {
    "candidate": "03_candidate_preference_standardized.jsonl",
    "sampling": "04_boundary_sampling_standardized.jsonl",
    "hidden": "05_hidden_state_capture_standardized.jsonl",
    "masking": "06_source_masking_standardized.jsonl",
}
ID_FIELDS = {
    "candidate": "candidate_record_id",
    "sampling": "sample_record_id",
    "hidden": "hidden_record_id",
    "masking": "mask_record_id",
}


def expected_ids(
    bundle_root: Path, config: dict[str, Any], model_key: str
) -> dict[str, set[str]]:
    rows = read_jsonl(
        bundle_root / "data" / "model_inputs" / f"{model_key}_diagnostic_inputs.jsonl"
    )
    full = [row for row in rows if row["condition"] == "full_cot_normal"]
    subset = set(
        read_json(bundle_root / config["candidate_preference"]["sensitivity_subset_file"])[
            "pair_ids"
        ]
    )
    primary = config["candidate_preference"]["primary_template"]
    sensitivity = config["candidate_preference"]["sensitivity_templates"]
    candidate = set()
    for row in full:
        candidate.add(
            f"{model_key}::{row['pair_id']}::full_cot_normal::{primary}"
        )
        if row["pair_id"] in subset:
            for template in sensitivity:
                candidate.add(
                    f"{model_key}::{row['pair_id']}::full_cot_normal::{template}"
                )
    sampling = {
        f"{model_key}::{row['pair_id']}::full_cot_normal::temp_{temperature}"
        f"::sample_{sample_index:02d}"
        for row in full
        for temperature in config["boundary_sampling"]["temperatures"]
        for sample_index in range(
            int(config["boundary_sampling"]["samples_per_temperature"])
        )
    }
    hidden = {
        f"{model_key}::{row['pair_id']}::{row['condition']}"
        "::answer_boundary_and_prefix"
        for row in rows
    }
    masking = {
        f"{model_key}::{row['pair_id']}::{row['condition']}::mask_{condition}"
        for row in rows
        for condition in config["source_masking"]["conditions"]
    }
    return {
        "candidate": candidate,
        "sampling": sampling,
        "hidden": hidden,
        "masking": masking,
    }


def exact_rows(
    path: Path, id_field: str, expected: set[str], problems: list[str]
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [str(row.get(id_field, "")) for row in rows]
    observed = set(ids)
    if len(ids) != len(observed):
        problems.append(f"{path}: duplicate {id_field} values")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        problems.append(f"{path}: {len(missing)} missing IDs; first={missing[:3]}")
    if extra:
        problems.append(f"{path}: {len(extra)} extra IDs; first={extra[:3]}")
    return rows


def validate_candidate(rows: list[dict[str, Any]], problems: list[str]) -> None:
    for row in rows:
        for branch in ("q_score", "qstar_score"):
            score = row.get(branch, {})
            values = score.get("token_logprobs", [])
            if not values:
                problems.append(f"{row.get('candidate_record_id')}: empty {branch}")
                continue
            if not all(math.isfinite(float(value)) for value in values):
                problems.append(f"{row.get('candidate_record_id')}: non-finite {branch}")
            if int(score.get("token_count", -1)) != len(values):
                problems.append(f"{row.get('candidate_record_id')}: {branch} token-count mismatch")
            for key in ("total_logprob", "average_logprob", "perplexity"):
                if not math.isfinite(float(score.get(key, float("nan")))):
                    problems.append(f"{row.get('candidate_record_id')}: invalid {branch}.{key}")
        if not row.get("q_candidate_token_ids") or not row.get("qstar_candidate_token_ids"):
            problems.append(f"{row.get('candidate_record_id')}: missing candidate token IDs")


def validate_sampling(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    profile: dict[str, Any],
    problems: list[str],
) -> None:
    endpoint = int(config["boundary_sampling"]["first_endpoint_tokens"])
    seen_seeds = set()
    for row in rows:
        record_id = row.get("sample_record_id")
        expected_seed = stable_seed(
            profile["revision"], record_id, int(config["seeds"]["sampling"])
        )
        if int(row.get("seed", -1)) != expected_seed:
            problems.append(f"{record_id}: incorrect deterministic seed")
        seed_key = (record_id, int(row.get("seed", -1)))
        if seed_key in seen_seeds:
            problems.append(f"{record_id}: duplicate request/seed tuple")
        seen_seeds.add(seed_key)
        generated = row.get("generated_token_ids", [])
        answer = row.get("final_answer_token_ids", [])
        endpoint_ids = row.get("first_endpoint_token_ids", [])
        if not generated:
            problems.append(f"{record_id}: empty generated IDs")
        if endpoint_ids != answer[:endpoint]:
            problems.append(f"{record_id}: first-{endpoint} endpoint mismatch")
        if int(row.get("first_endpoint_token_limit", -1)) != endpoint:
            problems.append(f"{record_id}: endpoint limit mismatch")
        if "backend_finish_reason" not in row or "backend_stop_reason" not in row:
            problems.append(f"{record_id}: missing backend stop provenance")
        if row.get("temperature") not in config["boundary_sampling"]["temperatures"]:
            problems.append(f"{record_id}: unregistered temperature")


def validate_hidden(
    rows: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    problems: list[str],
) -> None:
    registered_k = config["hidden_state_capture"]["answer_prefix_positions"]
    for row in rows:
        record_id = row.get("hidden_record_id")
        npz_path = output_root / row.get("npz_file", "")
        if not npz_path.exists():
            problems.append(f"{record_id}: missing NPZ archive")
            continue
        if sha256_file(npz_path) != row.get("npz_sha256"):
            problems.append(f"{record_id}: NPZ hash mismatch")
            continue
        try:
            with np.load(npz_path) as archive:
                hidden = archive["hidden"]
                positions = archive["absolute_positions"].tolist()
                k_values = archive["answer_prefix_positions"].tolist()
                input_ids = archive["input_token_ids"]
        except Exception as exc:
            problems.append(f"{record_id}: unreadable NPZ ({exc})")
            continue
        if list(hidden.shape) != row.get("hidden_shape"):
            problems.append(f"{record_id}: hidden shape mismatch")
        if str(hidden.dtype) != row.get("hidden_dtype"):
            problems.append(f"{record_id}: hidden dtype mismatch")
        if not np.isfinite(hidden).all() or not row.get("all_finite"):
            problems.append(f"{record_id}: non-finite hidden values")
        if positions != row.get("absolute_positions"):
            problems.append(f"{record_id}: absolute-position mismatch")
        if k_values != registered_k:
            problems.append(f"{record_id}: answer-prefix positions are not registered")
        if len(input_ids) != int(row.get("captured_input_token_count", -1)):
            problems.append(f"{record_id}: captured input length mismatch")


def validate_masking(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    problems: list[str],
    warnings: list[str],
) -> None:
    endpoint = int(config["source_masking"]["first_endpoint_tokens"])
    allowed = set(config["source_masking"]["conditions"])
    by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        record_id = row.get("mask_record_id")
        condition = row.get("mask_condition")
        if condition not in allowed:
            problems.append(f"{record_id}: unregistered mask condition")
            continue
        by_parent.setdefault(row["diagnostic_parent_id"], {})[condition] = row
        provenance = row.get("mask_provenance", {})
        if not provenance.get("initial_prefix_mask_applied"):
            problems.append(f"{record_id}: mask absent from initial prefix pass")
        if provenance.get("mask_condition") != condition:
            problems.append(f"{record_id}: mask provenance condition mismatch")
        if not provenance.get("prefix_mask_sha256"):
            problems.append(f"{record_id}: missing prefix-mask hash")
        masked_count = int(provenance.get("masked_position_count", -1))
        if condition == "none" and masked_count != 0:
            problems.append(f"{record_id}: unmasked control masks positions")
        if condition != "none" and masked_count <= 0:
            problems.append(f"{record_id}: selected source mask is empty")
        answer = row.get("final_answer_token_ids", [])
        if row.get("first_endpoint_token_ids") != answer[:endpoint]:
            problems.append(f"{record_id}: first-{endpoint} endpoint mismatch")
    for parent_id, variants in by_parent.items():
        if set(variants) != allowed:
            problems.append(f"{parent_id}: incomplete four-condition mask family")
            continue
        q_count = variants["question"]["mask_provenance"]["masked_position_count"]
        t_count = variants["trace"]["mask_provenance"]["masked_position_count"]
        j_count = variants["joint"]["mask_provenance"]["masked_position_count"]
        if j_count < max(q_count, t_count):
            problems.append(f"{parent_id}: joint mask is smaller than a component mask")
        none = variants["none"]
        if (
            none.get("unmasked_control_classifiable")
            and none.get("unmasked_control_branch_match") is not True
        ):
            warnings.append(
                f"{parent_id}: unmasked control changed coarse branch; retained as diagnostic drift, not a collection failure"
            )


def validate_manifests(
    output_root: Path, model_key: str, problems: list[str]
) -> None:
    root = output_root / "models" / model_key / "manifests"
    required = [
        "candidate_preflight.json",
        "hidden_preflight.json",
        "masking_preflight.json",
        "sampling_preflight.json",
        "candidate_stage_manifest.json",
        "hidden_stage_manifest.json",
        "masking_stage_manifest.json",
        "sampling_stage_manifest.json",
        "transformers_runtime.json",
        "vllm_sampling_runtime.json",
    ]
    for name in required:
        path = root / name
        if not path.exists():
            problems.append(f"{model_key}: missing manifest {name}")
            continue
        if name.endswith("_stage_manifest.json"):
            payload = read_json(path)
            if not payload.get("complete"):
                problems.append(f"{model_key}: incomplete stage manifest {name}")


def validate_model(
    bundle_root: Path,
    output_root: Path,
    config: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    expected = expected_ids(bundle_root, config, model_key)
    root = output_root / "models" / model_key
    problems: list[str] = []
    warnings: list[str] = []
    rows = {}
    for stage, filename in CHECKPOINTS.items():
        rows[stage] = exact_rows(
            root / "checkpoints" / filename,
            ID_FIELDS[stage],
            expected[stage],
            problems,
        )
    validate_candidate(rows["candidate"], problems)
    validate_sampling(
        rows["sampling"], config, config["models"][model_key], problems
    )
    validate_hidden(rows["hidden"], output_root, config, problems)
    validate_masking(rows["masking"], config, problems, warnings)
    validate_manifests(output_root, model_key, problems)
    summary = {
        "created_at_utc": utc_now(),
        "model_key": model_key,
        "expected": {key: len(value) for key, value in expected.items()},
        "observed": {key: len(value) for key, value in rows.items()},
        "complete": not problems,
        "problems": problems,
        "warnings": warnings,
    }
    atomic_write_json(
        root / "manifests" / "diagnostic_quality_gate_summary.json", summary
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key")
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(
        bundle_root / "config" / "standardized_diagnostics_config.json"
    )
    models = [args.model_key] if args.model_key else list(config["models"])
    summaries = {
        model_key: validate_model(bundle_root, output_root, config, model_key)
        for model_key in models
    }
    all_complete = all(value["complete"] for value in summaries.values())
    aggregate = {
        "created_at_utc": utc_now(),
        "study_id": config["study_id"],
        "cohort_size": len(read_jsonl(bundle_root / config["cohort_file"])),
        "all_complete": all_complete,
        "models": summaries,
    }
    atomic_write_json(output_root / "diagnostic_quality_gate_summary.json", aggregate)
    for model_key, summary in summaries.items():
        label = "OK" if summary["complete"] else "FAILED"
        print(f"[{label}] {model_key}: {summary['observed']}")
        for problem in summary["problems"][:20]:
            print("  -", problem)
        for warning in summary.get("warnings", [])[:10]:
            print("  warning -", warning)
    if not all_complete:
        raise SystemExit("Diagnostic quality gate failed; see diagnostic_quality_gate_summary.json")


if __name__ == "__main__":
    main()
