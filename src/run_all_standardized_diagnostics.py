from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import atomic_write_json, read_json, read_jsonl, sha256_file, utc_now


EXPECTED = {"candidate": 65, "sampling": 696, "hidden": 87, "masking": 348}
CHECKPOINTS = {
    "candidate": "03_candidate_preference_standardized.jsonl",
    "sampling": "04_boundary_sampling_standardized.jsonl",
    "hidden": "05_hidden_state_capture_standardized.jsonl",
    "masking": "06_source_masking_standardized.jsonl",
}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def collect_counts(
    output_root: Path, models: list[str]
) -> dict[str, dict[str, int]]:
    counts = {}
    for model_key in models:
        checkpoint_dir = output_root / "models" / model_key / "checkpoints"
        counts[model_key] = {
            stage: count_jsonl(checkpoint_dir / filename)
            for stage, filename in CHECKPOINTS.items()
        }
    return counts


def infer_stage(counts: dict[str, int], stages: list[str]) -> str:
    for stage in stages:
        if stage in EXPECTED and counts[stage] < EXPECTED[stage]:
            return stage
    return stages[-1] if stages else "quality_gate"


def write_progress(
    output_root: Path,
    models: list[str],
    status: str,
    current_model: str | None,
    stages: list[str],
    heartbeat_seconds: int,
    command: list[str] | None = None,
    message: str | None = None,
) -> None:
    counts = collect_counts(output_root, models)
    model_summaries = {}
    global_completed = 0
    global_expected = len(models) * sum(EXPECTED.values())
    for model_key in models:
        stage_summary = {}
        for stage, expected in EXPECTED.items():
            observed = counts[model_key][stage]
            completed = min(observed, expected)
            global_completed += completed
            stage_summary[stage] = {
                "completed": observed,
                "expected": expected,
                "percent": round(100.0 * completed / expected, 2),
                "complete": observed == expected,
            }
        model_summaries[model_key] = stage_summary
    current_stage = None
    if current_model and current_model in counts:
        current_stage = infer_stage(counts[current_model], stages)
    atomic_write_json(
        output_root / "logs" / "progress.json",
        {
            "updated_at_utc": utc_now(),
            "heartbeat_seconds": heartbeat_seconds,
            "status": status,
            "current_model": current_model,
            "current_stage": current_stage,
            "current_stage_completed": (
                counts.get(current_model, {}).get(current_stage)
                if current_stage
                else None
            ),
            "current_stage_expected": EXPECTED.get(current_stage),
            "global_completed": global_completed,
            "global_expected": global_expected,
            "global_percent": round(100.0 * global_completed / global_expected, 2),
            "models": model_summaries,
            "command": [str(part) for part in command] if command else None,
            "message": message,
        },
    )


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_latest_error(
    output_root: Path, error: BaseException, command: list[str] | None
) -> None:
    log_path = output_root / "logs" / "run.log"
    tail = ""
    if log_path.exists():
        tail = "".join(
            log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)[-120:]
        )
    error_path = output_root / "logs" / "latest_error.txt"
    temporary = error_path.with_suffix(".tmp")
    temporary.write_text(
        f"Timestamp (UTC): {utc_now()}\n"
        f"Error type: {type(error).__name__}\n"
        f"Error: {error}\n"
        f"Command: {' '.join(str(part) for part in command) if command else 'orchestrator'}\n\n"
        f"Traceback:\n{traceback.format_exc()}\n"
        f"Last 120 run.log lines:\n{tail}",
        encoding="utf-8",
    )
    os.replace(temporary, error_path)


def run(
    command: list[Any],
    output_root: Path,
    models: list[str],
    model_key: str | None,
    stages: list[str],
    heartbeat_seconds: int,
) -> None:
    cmd = [str(part) for part in command]
    log_path = output_root / "logs" / "run.log"
    display = " ".join(cmd)
    print("\n>>>", display, flush=True)
    append_log(log_path, f"START {display}")
    write_progress(
        output_root,
        models,
        "running",
        model_key,
        stages,
        heartbeat_seconds,
        cmd,
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT
        )
        while True:
            try:
                return_code = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                log_handle.flush()
                os.fsync(log_handle.fileno())
                write_progress(
                    output_root,
                    models,
                    "running",
                    model_key,
                    stages,
                    heartbeat_seconds,
                    cmd,
                )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    write_progress(
        output_root,
        models,
        "running",
        model_key,
        stages,
        heartbeat_seconds,
        cmd,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    append_log(log_path, f"DONE {display}")


def copy_frozen_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(path) != sha256_file(target):
                raise RuntimeError(
                    f"Frozen output conflict: {target} differs from bundle input"
                )
        else:
            shutil.copy2(path, target)


def preserve_inputs(bundle_root: Path, output_root: Path) -> dict[str, Any]:
    copy_frozen_tree(bundle_root / "config", output_root / "frozen_inputs" / "config")
    copy_frozen_tree(bundle_root / "data", output_root / "frozen_inputs" / "data")
    copy_frozen_tree(bundle_root / "existing_evidence", output_root / "existing_evidence")
    copy_frozen_tree(bundle_root / "plans", output_root / "plans")
    for name in ("README.md", "requirements.txt", "PACKAGE_PLAN_CROSSCHECK.md", "PACKAGE_PLAN_CROSSCHECK.json"):
        source = bundle_root / name
        if source.exists():
            target = output_root / "frozen_inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256_file(source) != sha256_file(target):
                raise RuntimeError(f"Frozen output conflict: {target}")
            if not target.exists():
                shutil.copy2(source, target)
    files = []
    for root_name in ("frozen_inputs", "existing_evidence", "plans"):
        for path in sorted((output_root / root_name).rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(output_root)).replace("\\", "/"),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    manifest = {
        "created_at_utc": utc_now(),
        "immutable_inputs_preserved": True,
        "files": files,
    }
    atomic_write_json(output_root / "FROZEN_INPUT_MANIFEST.json", manifest)
    return manifest


def validate_frozen_bundle(bundle_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cohort = read_jsonl(bundle_root / config["cohort_file"])
    pair_ids = {row["pair_id"] for row in cohort}
    if len(cohort) != 29 or len(pair_ids) != 29:
        raise RuntimeError("Frozen common cohort must contain 29 unique pairs")
    if "nat_lin3_008" in pair_ids:
        raise RuntimeError("Outcome-independent source-ineligible pair nat_lin3_008 was not excluded")
    conditions = set(config["diagnostic_conditions"])
    model_inputs = {}
    for model_key in config["models"]:
        rows = read_jsonl(
            bundle_root
            / "data"
            / "model_inputs"
            / f"{model_key}_diagnostic_inputs.jsonl"
        )
        observed_pairs = {row["pair_id"] for row in rows}
        observed_conditions = {row["condition"] for row in rows}
        parent_ids = {row["diagnostic_parent_id"] for row in rows}
        if len(rows) != 87 or len(parent_ids) != 87:
            raise RuntimeError(f"{model_key}: expected 87 unique diagnostic parents")
        if observed_pairs != pair_ids:
            raise RuntimeError(f"{model_key}: pair IDs differ from the common cohort")
        if observed_conditions != conditions:
            raise RuntimeError(f"{model_key}: diagnostic conditions differ from the frozen design")
        model_inputs[model_key] = {
            "rows": len(rows),
            "unique_pairs": len(observed_pairs),
            "conditions": sorted(observed_conditions),
        }
    subset = read_json(
        bundle_root / config["candidate_preference"]["sensitivity_subset_file"]
    )["pair_ids"]
    if len(subset) != 9 or not set(subset).issubset(pair_ids):
        raise RuntimeError("The fixed template-sensitivity subset must contain nine cohort pairs")
    evidence_manifest = read_json(
        bundle_root / "existing_evidence" / "canonical_all_models_manifest.json"
    )
    evidence_path = (
        bundle_root
        / "existing_evidence"
        / "canonical_all_models_annotations.jsonl"
    )
    expected_hash = evidence_manifest["outputs"]["canonical_annotations"]["sha256"]
    if (
        evidence_manifest.get("validation_status") != "PASS"
        or int(evidence_manifest.get("observed_rows", -1)) != 6231
        or count_jsonl(evidence_path) != 6231
        or sha256_file(evidence_path) != expected_hash
    ):
        raise RuntimeError("Preserved canonical behavioural evidence failed validation")
    return {
        "cohort_pairs": len(pair_ids),
        "excluded_pair_absent": True,
        "template_sensitivity_pairs": len(subset),
        "model_inputs": model_inputs,
        "canonical_annotation_rows": 6231,
        "canonical_annotation_sha256": expected_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma_e2b", "gpt_oss_20b", "qwen35_9b"],
    )
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = read_json(
        bundle_root / "config" / "standardized_diagnostics_config.json"
    )
    if args.models != list(config["models"]):
        raise RuntimeError(
            "The final run must use the frozen model order: "
            + " ".join(config["models"])
        )
    bundle_validation = validate_frozen_bundle(bundle_root, config)
    heartbeat_seconds = int(config["runtime"]["heartbeat_seconds"])
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_error = log_dir / "latest_error.txt"
    if latest_error.exists():
        latest_error.unlink()
    frozen_manifest = preserve_inputs(bundle_root, output_root)
    atomic_write_json(
        output_root / "run_manifest.json",
        {
            "created_at_utc": utc_now(),
            "bundle_root": str(bundle_root),
            "output_root": str(output_root),
            "model_order": args.models,
            "study_id": config["study_id"],
            "study_version": config["study_version"],
            "config_sha256": sha256_file(
                bundle_root / "config" / "standardized_diagnostics_config.json"
            ),
            "heartbeat_seconds": heartbeat_seconds,
            "progress_file": "logs/progress.json",
            "run_log": "logs/run.log",
            "latest_error_file": "logs/latest_error.txt",
            "frozen_input_file_count": len(frozen_manifest["files"]),
            "bundle_validation": bundle_validation,
            "resume_policy": "append-only JSONL checkpoints with exact-ID gates",
        },
    )

    active_command: list[Any] | None = None
    active_model: str | None = None
    active_stages: list[str] = []
    try:
        write_progress(
            output_root,
            args.models,
            "starting",
            None,
            [],
            heartbeat_seconds,
            message="Preparing the final standardized three-model diagnostics",
        )
        for model_key in args.models:
            active_model = model_key
            active_stages = ["candidate", "hidden", "masking"]
            active_command = [
                sys.executable,
                bundle_root / "src" / "run_transformers_diagnostics.py",
                "--bundle-root",
                bundle_root,
                "--output-root",
                output_root,
                "--model-key",
                model_key,
                "--stages",
                *active_stages,
            ]
            run(
                active_command,
                output_root,
                args.models,
                model_key,
                active_stages,
                heartbeat_seconds,
            )

            active_stages = ["sampling"]
            active_command = [
                sys.executable,
                bundle_root / "src" / "run_vllm_sampling.py",
                "--bundle-root",
                bundle_root,
                "--output-root",
                output_root,
                "--model-key",
                model_key,
            ]
            run(
                active_command,
                output_root,
                args.models,
                model_key,
                active_stages,
                heartbeat_seconds,
            )

            active_stages = ["quality_gate"]
            active_command = [
                sys.executable,
                bundle_root / "src" / "quality_gate.py",
                "--bundle-root",
                bundle_root,
                "--output-root",
                output_root,
                "--model-key",
                model_key,
            ]
            run(
                active_command,
                output_root,
                args.models,
                model_key,
                active_stages,
                heartbeat_seconds,
            )

        active_model = None
        active_stages = ["quality_gate"]
        active_command = [
            sys.executable,
            bundle_root / "src" / "quality_gate.py",
            "--bundle-root",
            bundle_root,
            "--output-root",
            output_root,
        ]
        run(
            active_command,
            output_root,
            args.models,
            None,
            active_stages,
            heartbeat_seconds,
        )

        active_stages = ["finalize"]
        active_command = [
            sys.executable,
            bundle_root / "src" / "finalize_analysis_ready_collection.py",
            "--bundle-root",
            bundle_root,
            "--output-root",
            output_root,
        ]
        run(
            active_command,
            output_root,
            args.models,
            None,
            active_stages,
            heartbeat_seconds,
        )
        write_progress(
            output_root,
            args.models,
            "completed",
            None,
            [],
            heartbeat_seconds,
            message=(
                "Final model collection complete. Manual diagnostic coding is "
                "the only remaining analysis-readiness step."
            ),
        )
        append_log(log_dir / "run.log", "FINAL MODEL COLLECTION COMPLETED SUCCESSFULLY")
    except BaseException as error:
        write_latest_error(output_root, error, active_command)
        append_log(
            log_dir / "run.log",
            f"RUN FAILED: {type(error).__name__}: {error}",
        )
        write_progress(
            output_root,
            args.models,
            "failed",
            active_model,
            active_stages,
            heartbeat_seconds,
            active_command,
            f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
