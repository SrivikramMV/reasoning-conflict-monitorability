

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import atomic_write_json, read_json, sha256_file, utc_now


def log(path: Path, message: str) -> None:
    line = f"{utc_now()} [ORCHESTRATOR] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def verify_bundle(bundle_root: Path) -> dict[str, Any]:
    manifest = read_json(bundle_root / "bundle_checksums.json")
    failures = []
    for relative, expected in manifest["files"].items():
        path = bundle_root / relative
        if not path.exists():
            failures.append(f"missing {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"checksum mismatch {relative}")
    if failures:
        raise RuntimeError("Bundle verification failed: " + "; ".join(failures))
    return manifest


def initialise_frozen_output(bundle_root: Path, output_root: Path) -> dict[str, Any]:
    config = read_json(bundle_root / "config" / "cross_model_study.json")
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    specification = {
        "study_id": config["study_id"],
        "study_version": config["study_version"],
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(bundle_root / "config" / "cross_model_study.json"),
        "benchmark_sha256": sha256_file(bundle_root / config["benchmark_file"]),
        "stability_subset_sha256": sha256_file(bundle_root / config["stability_subset_file"]),
        "gemma_source_sha256": sha256_file(bundle_root / config["gemma_source_file"]),
        "models": config["models"],
        "expected_rows": config["expected_rows"],
    }
    frozen_path = manifests / "frozen_run_specification.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        for key in (
            "study_id",
            "study_version",
            "config_sha256",
            "benchmark_sha256",
            "stability_subset_sha256",
            "gemma_source_sha256",
            "models",
            "expected_rows",
        ):
            if existing.get(key) != specification.get(key):
                raise RuntimeError(
                    f"Drive output contains an incompatible frozen run specification ({key})"
                )
    else:
        atomic_write_json(frozen_path, specification)
        copies = {
            bundle_root / "config" / "cross_model_study.json": manifests
            / "cross_model_study.json",
            bundle_root / config["benchmark_file"]: manifests / "frozen_natural_pairs_120.jsonl",
            bundle_root / config["stability_subset_file"]: manifests / "stability_subset_30.json",
            bundle_root / "data" / "benchmark_audit.md": manifests / "benchmark_audit.md",
            bundle_root / "data" / "data_manifest.json": manifests / "data_manifest.json",
            bundle_root / "STUDY_SPECIFICATION.md": manifests / "STUDY_SPECIFICATION.md",
            bundle_root / "rubrics" / "taxonomy_and_transparency_protocol.md": manifests
            / "taxonomy_and_transparency_protocol.md",
        }
        for source, target in copies.items():
            shutil.copy2(source, target)
    return config


def run_child(command: list[str], transcript: Path, env: dict[str, str]) -> None:
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{utc_now()} [COMMAND] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def count_rows(output_root: Path) -> int:
    total = 0
    for path in output_root.glob("models/*/checkpoints/*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def build_final_archive(output_root: Path, config: dict[str, Any]) -> Path:
    archive_base = output_root.parent / f"{config['study_id']}_results"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_root.parent,
            base_dir=output_root.name,
        )
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "RUN_FAILED.json").unlink(missing_ok=True)
    transcript = output_root / "logs" / "orchestrator.log"
    try:
        verify_bundle(bundle_root)
        config = initialise_frozen_output(bundle_root, output_root)
        log(transcript, f"Frozen study {config['study_id']} verified")
        env = dict(os.environ)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        stages = [
            (
                "gemma_e2b",
                [
                    sys.executable,
                    str(bundle_root / "src" / "run_gemma_remaining.py"),
                    "--bundle-root",
                    str(bundle_root),
                    "--output-root",
                    str(output_root),
                ],
            ),
            (
                "qwen35_9b",
                [
                    sys.executable,
                    str(bundle_root / "src" / "run_replication_model.py"),
                    "--bundle-root",
                    str(bundle_root),
                    "--output-root",
                    str(output_root),
                    "--model-key",
                    "qwen35_9b",
                ],
            ),
            (
                "gpt_oss_20b",
                [
                    sys.executable,
                    str(bundle_root / "src" / "run_replication_model.py"),
                    "--bundle-root",
                    str(bundle_root),
                    "--output-root",
                    str(output_root),
                    "--model-key",
                    "gpt_oss_20b",
                ],
            ),
        ]
        for model_key, command in stages:
            marker = output_root / "models" / model_key / "MODEL_RUN_COMPLETE.json"
            gate_path = output_root / "models" / model_key / "QUALITY_GATE.json"
            if marker.exists() and gate_path.exists() and read_json(gate_path).get("passed"):
                log(transcript, f"Skipping completed model {model_key}")
                continue
            log(transcript, f"Starting model {model_key}")
            run_child(command, transcript, env)
            log(transcript, f"Finished model {model_key}; {count_rows(output_root)} rows on Drive")
        observed = count_rows(output_root)
        expected = int(config["expected_rows"]["new_rows_total"])
        gates = {
            key: read_json(output_root / "models" / key / "QUALITY_GATE.json")
            for key in ("gemma_e2b", "qwen35_9b", "gpt_oss_20b")
        }
        failures = [key for key, gate in gates.items() if not gate.get("passed")]
        if observed != expected:
            failures.append(f"row_count:{observed}!={expected}")
        overall = {
            "created_at_utc": utc_now(),
            "study_id": config["study_id"],
            "passed": not failures,
            "expected_rows": expected,
            "observed_rows": observed,
            "failures": failures,
            "model_quality_gates": gates,
        }
        atomic_write_json(output_root / "OVERALL_QUALITY_GATE.json", overall)
        if failures:
            raise RuntimeError(f"Overall quality gate failed: {failures}")
        atomic_write_json(
            output_root / "RUN_COMPLETE.json",
            {
                "completed_at_utc": utc_now(),
                "study_id": config["study_id"],
                "observed_rows": observed,
            },
        )
        archive = build_final_archive(output_root, config)
        atomic_write_json(
            output_root / "RESULT_ARCHIVE.json",
            {
                "created_at_utc": utc_now(),
                "archive_path": str(archive),
                "archive_sha256": sha256_file(archive),
            },
        )
        log(transcript, f"All stages complete. Final archive: {archive}")
    except Exception as exc:
        atomic_write_json(
            output_root / "RUN_FAILED.json",
            {
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "completed_rows_retained": count_rows(output_root),
            },
        )
        log(transcript, f"Run failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
