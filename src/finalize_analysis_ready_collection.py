from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


CHECKPOINTS = {
    "candidate": "03_candidate_preference_standardized.jsonl",
    "sampling": "04_boundary_sampling_standardized.jsonl",
    "hidden": "05_hidden_state_capture_standardized.jsonl",
    "masking": "06_source_masking_standardized.jsonl",
}


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def review_record(
    family: str,
    row: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    record_id = (
        row["sample_record_id"] if family == "sampling" else row["mask_record_id"]
    )
    return {
        "review_record_id": f"{family}::{record_id}",
        "record_family": family,
        "result_record_id": record_id,
        "model_key": row["model_key"],
        "pair_id": row["pair_id"],
        "category": row["category"],
        "condition": row["condition"],
        "temperature": row.get("temperature"),
        "sample_index": row.get("sample_index"),
        "mask_condition": row.get("mask_condition"),
        "original_question": parent["original_question"],
        "counterfactual_question": parent["counterfactual_question"],
        "original_answer": parent["original_answer"],
        "counterfactual_answer": parent["counterfactual_answer"],
        "prefilled_or_completed_reasoning_trace": parent["thought"],
        "parent_natural_final_answer": parent["final_answer"],
        "diagnostic_generated_final_answer": row["final_answer"],
        "first_192_token_endpoint": row["first_endpoint_text"],
        "automatic_visible_branch": row.get("automatic_visible_branch"),
        "automatic_final_branch": row.get("automatic_final_branch"),
        "technical_usability": None,
        "manual_reconstructed_branch": None,
        "manual_final_answer_state": None,
        "manual_visible_conflict_or_correction": None,
        "manual_taxonomy_or_diagnostic_label": None,
        "manual_confidence": None,
        "manual_evidence_excerpt": None,
        "manual_rationale": None,
        "reviewer": None,
        "reviewed_at_utc": None,
    }


def file_inventory(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and ".tmp" not in path.name:
            rows.append(
                {
                    "path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(
        bundle_root / "config" / "standardized_diagnostics_config.json"
    )
    quality = read_json(output_root / "diagnostic_quality_gate_summary.json")
    if not quality.get("all_complete"):
        raise RuntimeError("Cannot finalize a collection that failed the quality gate")

    final_dir = output_root / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)
    parent_lookup = {}
    all_results: dict[str, list[dict[str, Any]]] = {
        key: [] for key in CHECKPOINTS
    }
    for model_key in config["models"]:
        parents = read_jsonl(
            bundle_root
            / "data"
            / "model_inputs"
            / f"{model_key}_diagnostic_inputs.jsonl"
        )
        parent_lookup.update(
            {row["diagnostic_parent_id"]: row for row in parents}
        )
        checkpoint_dir = output_root / "models" / model_key / "checkpoints"
        for family, filename in CHECKPOINTS.items():
            rows = read_jsonl(checkpoint_dir / filename)
            all_results[family].extend(rows)
            destination = final_dir / f"{model_key}__{family}.jsonl"
            shutil.copy2(checkpoint_dir / filename, destination)

    combined_paths = {}
    for family, rows in all_results.items():
        path = final_dir / f"all_models__{family}.jsonl"
        atomic_write_jsonl(path, rows)
        combined_paths[family] = {
            "path": str(path.relative_to(output_root)).replace("\\", "/"),
            "records": len(rows),
            "sha256": sha256_file(path),
        }

    review_rows = []
    for family in ("sampling", "masking"):
        for row in all_results[family]:
            review_rows.append(
                review_record(
                    family,
                    row,
                    parent_lookup[row["diagnostic_parent_id"]],
                )
            )
    review_rows.sort(
        key=lambda row: (
            row["model_key"],
            row["record_family"],
            row["pair_id"],
            row["condition"],
            str(row.get("temperature")),
            -1 if row.get("sample_index") is None else row["sample_index"],
            str(row.get("mask_condition")),
        )
    )
    queue_path = final_dir / "manual_review_queue.jsonl"
    atomic_write_jsonl(queue_path, review_rows)
    if len(review_rows) != int(config["runtime"]["manual_review_rows"]):
        raise RuntimeError(
            f"Manual-review queue has {len(review_rows)} rows; expected "
            f"{config['runtime']['manual_review_rows']}"
        )

    existing_annotation = (
        output_root
        / "existing_evidence"
        / "canonical_all_models_annotations.jsonl"
    )
    if not existing_annotation.exists():
        raise RuntimeError("Canonical behavioural annotation evidence was not preserved")

    manifest = {
        "created_at_utc": utc_now(),
        "study_id": config["study_id"],
        "study_version": config["study_version"],
        "collection_complete": True,
        "analysis_ready": False,
        "analysis_ready_blocker": (
            "Manual coding of the registered sampling and source-masking "
            "continuations remains."
        ),
        "models": list(config["models"]),
        "common_diagnostic_cohort_size": 29,
        "diagnostic_units": {
            "candidate": len(all_results["candidate"]),
            "sampling": len(all_results["sampling"]),
            "hidden": len(all_results["hidden"]),
            "masking": len(all_results["masking"]),
            "total": sum(len(value) for value in all_results.values()),
        },
        "manual_review": {
            "queue_path": str(queue_path.relative_to(output_root)).replace("\\", "/"),
            "rows": len(review_rows),
            "sha256": sha256_file(queue_path),
            "required_before_final_analysis": True,
        },
        "existing_behavioural_evidence": {
            "path": str(existing_annotation.relative_to(output_root)).replace("\\", "/"),
            "records": len(read_jsonl(existing_annotation)),
            "sha256": sha256_file(existing_annotation),
            "status": "retained_final_evidence",
        },
        "combined_result_files": combined_paths,
        "quality_gate_path": "diagnostic_quality_gate_summary.json",
        "quality_gate_sha256": sha256_file(
            output_root / "diagnostic_quality_gate_summary.json"
        ),
        "no_further_model_generation_required": True,
        "excluded_from_stopping_point": [
            "activation patching",
            "additional model families",
            "new benchmark questions",
            "new injection depths",
            "fine-tuning",
        ],
    }
    atomic_write_json(output_root / "ANALYSIS_MANIFEST.json", manifest)
    marker = {
        "created_at_utc": utc_now(),
        "study_id": config["study_id"],
        "status": "FINAL_MODEL_COLLECTION_COMPLETE",
        "collection_complete": True,
        "analysis_ready": False,
        "remaining_non_generation_work": [
            "manual coding of 3,132 diagnostic continuations",
            "freeze merged manual annotations",
            "final statistical and qualitative analysis",
            "writing",
        ],
        "no_further_model_run_is_registered": True,
        "analysis_manifest": "ANALYSIS_MANIFEST.json",
    }
    atomic_write_json(output_root / "FINAL_COLLECTION_COMPLETE.json", marker)
    inventory = file_inventory(output_root)
    atomic_write_json(
        output_root / "FILE_INVENTORY.json",
        {
            "created_at_utc": utc_now(),
            "files": inventory,
            "file_count": len(inventory),
        },
    )
    print("Final model collection complete.")
    print(f"Diagnostic units: {manifest['diagnostic_units']['total']}")
    print(f"Manual-review rows: {len(review_rows)}")
    print("No additional model generation is registered.")


if __name__ == "__main__":
    main()
