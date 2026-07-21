from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import atomic_write_json, read_json, utc_now


EXPECTED = {"candidate": 65, "sampling": 696, "hidden": 87, "masking": 348}
CHECKPOINTS = {
    "candidate": "03_candidate_preference_standardized.jsonl",
    "sampling": "04_boundary_sampling_standardized.jsonl",
    "hidden": "05_hidden_state_capture_standardized.jsonl",
    "masking": "06_source_masking_standardized.jsonl",
}
HEARTBEAT_SECONDS = 30


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def collect_counts(output_root: Path, models: list[str]) -> dict[str, dict[str, int]]:
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
            completed = min(counts[model_key][stage], expected)
            global_completed += completed
            stage_summary[stage] = {
                "completed": completed,
                "expected": expected,
                "percent": round(100.0 * completed / expected, 2),
            }
        model_summaries[model_key] = stage_summary
    current_stage = None
    if current_model and current_model in counts:
        current_stage = infer_stage(counts[current_model], stages)
    atomic_write_json(output_root / "logs" / "progress.json", {
        "updated_at_utc": utc_now(),
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "status": status,
        "current_model": current_model,
        "current_stage": current_stage,
        "current_stage_completed": counts.get(current_model, {}).get(current_stage) if current_stage else None,
        "current_stage_expected": EXPECTED.get(current_stage) if current_stage else None,
        "global_completed": global_completed,
        "global_expected": global_expected,
        "global_percent": round(100.0 * global_completed / global_expected, 2),
        "models": model_summaries,
        "command": [str(part) for part in command] if command else None,
        "message": message,
    })


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_latest_error(output_root: Path, error: BaseException, command: list[str] | None) -> None:
    log_path = output_root / "logs" / "run.log"
    tail = ""
    if log_path.exists():
        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-120:])
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
    cmd: list[str],
    output_root: Path,
    models: list[str],
    model_key: str | None,
    stages: list[str],
) -> None:
    command = [str(part) for part in cmd]
    log_path = output_root / "logs" / "run.log"
    display = " ".join(command)
    print("\n>>>", display, flush=True)
    append_log(log_path, f"START {display}")
    write_progress(output_root, models, "running", model_key, stages, command)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        while True:
            try:
                return_code = process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                log_handle.flush()
                os.fsync(log_handle.fileno())
                write_progress(output_root, models, "running", model_key, stages, command)
        log_handle.flush()
        os.fsync(log_handle.fileno())
    write_progress(output_root, models, "running", model_key, stages, command)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    append_log(log_path, f"DONE {display}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gemma_e2b", "gpt_oss_20b", "qwen35_9b"])
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = read_json(bundle_root / "config" / "standardized_diagnostics_config.json")
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_error = log_dir / "latest_error.txt"
    if latest_error.exists():
        latest_error.unlink()
    atomic_write_json(output_root / "run_manifest.json", {
        "created_at_utc": utc_now(),
        "bundle_root": str(bundle_root),
        "model_order": args.models,
        "study_id": config["study_id"],
        "study_version": config["study_version"],
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "progress_file": "logs/progress.json",
        "run_log": "logs/run.log",
        "latest_error_file": "logs/latest_error.txt",
    })
    active_command = None
    active_model = None
    active_stages = []
    try:
        write_progress(output_root, args.models, "starting", None, [], message="Preparing the standardized diagnostic run")
        for model_key in args.models:
            active_model = model_key
            if model_key == "gemma_e2b":
                stages = ["candidate", "sampling", "hidden", "masking"]
                active_stages = stages
                active_command = [sys.executable, bundle_root / "src" / "run_transformers_diagnostics.py", "--bundle-root", bundle_root, "--output-root", output_root, "--model-key", model_key, "--stages", *stages]
                run(active_command, output_root, args.models, model_key, stages)
            else:
                stages = ["candidate", "hidden", "masking"]
                active_stages = stages
                active_command = [sys.executable, bundle_root / "src" / "run_transformers_diagnostics.py", "--bundle-root", bundle_root, "--output-root", output_root, "--model-key", model_key, "--stages", *stages]
                run(active_command, output_root, args.models, model_key, stages)
                active_stages = ["sampling"]
                active_command = [sys.executable, bundle_root / "src" / "run_vllm_sampling.py", "--bundle-root", bundle_root, "--output-root", output_root, "--model-key", model_key]
                run(active_command, output_root, args.models, model_key, active_stages)
            active_stages = ["quality_gate"]
            active_command = [sys.executable, bundle_root / "src" / "quality_gate.py", "--bundle-root", bundle_root, "--output-root", output_root, "--model-key", model_key]
            run(active_command, output_root, args.models, model_key, active_stages)
        active_model = None
        active_stages = ["quality_gate"]
        active_command = [sys.executable, bundle_root / "src" / "quality_gate.py", "--bundle-root", bundle_root, "--output-root", output_root]
        run(active_command, output_root, args.models, None, active_stages)
        write_progress(output_root, args.models, "completed", None, [], message="All diagnostics and quality gates completed")
        append_log(log_dir / "run.log", "RUN COMPLETED SUCCESSFULLY")
    except BaseException as error:
        write_latest_error(output_root, error, active_command)
        append_log(log_dir / "run.log", f"RUN FAILED: {type(error).__name__}: {error}")
        write_progress(output_root, args.models, "failed", active_model, active_stages, active_command, f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
