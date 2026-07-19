

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import atomic_write_json, read_json, sha256_file, utc_now
from run_all import (
    build_final_archive,
    count_rows,
    initialise_frozen_output,
    log,
    run_child,
    verify_bundle,
)


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
        log(transcript, f"Focused completion study {config['study_id']} verified")
        env = dict(os.environ)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        stages = [
            (
                "gemma_e2b",
                bundle_root / "src" / "run_final_gemma_recovery.py",
            ),
            (
                "gpt_oss_20b",
                bundle_root / "src" / "run_final_gpt_recovery.py",
            ),
            (
                "qwen35_9b",
                bundle_root / "src" / "run_final_qwen.py",
            ),
        ]
        for model_key, script in stages:
            marker = output_root / "models" / model_key / "MODEL_RUN_COMPLETE.json"
            gate_path = output_root / "models" / model_key / "QUALITY_GATE.json"
            if marker.exists() and gate_path.exists() and read_json(gate_path).get("passed"):
                log(transcript, f"Skipping completed model {model_key}")
                continue
            command = [
                sys.executable,
                str(script),
                "--bundle-root",
                str(bundle_root),
                "--output-root",
                str(output_root),
            ]
            log(transcript, f"Starting focused stage for {model_key}")
            run_child(command, transcript, env)
            log(transcript, f"Finished {model_key}; {count_rows(output_root)} rows on Drive")

        model_keys = ("gemma_e2b", "gpt_oss_20b", "qwen35_9b")
        gates = {
            key: read_json(output_root / "models" / key / "QUALITY_GATE.json")
            for key in model_keys
        }
        observed = count_rows(output_root)
        expected = int(config["expected_rows"]["new_rows_total"])
        failures = [key for key, gate in gates.items() if not gate.get("passed")]
        if observed != expected:
            failures.append(f"row_count:{observed}!={expected}")
        analysis_ready = {
            key: bool(gate.get("analysis_ready", gate.get("passed")))
            for key, gate in gates.items()
        }
        overall = {
            "created_at_utc": utc_now(),
            "study_id": config["study_id"],
            "passed": not failures,
            "expected_rows": expected,
            "observed_rows": observed,
            "failures": failures,
            "analysis_ready_by_model": analysis_ready,
            "model_quality_gates": gates,
        }
        atomic_write_json(output_root / "OVERALL_QUALITY_GATE.json", overall)
        if failures:
            raise RuntimeError(f"Overall completion gate failed: {failures}")

        atomic_write_json(
            output_root / "RUN_COMPLETE.json",
            {
                "completed_at_utc": utc_now(),
                "study_id": config["study_id"],
                "observed_rows": observed,
                "analysis_ready_by_model": analysis_ready,
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
        log(transcript, f"Focused completion run finished. Archive: {archive}")
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
