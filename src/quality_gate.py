from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from common import atomic_write_json, read_json, read_jsonl, sha256_file, utc_now


def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key")
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    config = read_json(bundle_root / "config" / "standardized_diagnostics_config.json")
    cohort_n = len(read_jsonl(bundle_root / config["cohort_file"]))
    template_n = len(read_json(bundle_root / config["candidate_preference"]["sensitivity_subset_file"])["pair_ids"])
    expected = {
        "candidate": cohort_n + template_n * len(config["candidate_preference"]["sensitivity_templates"]),
        "sampling": cohort_n * len(config["boundary_sampling"]["temperatures"]) * int(config["boundary_sampling"]["samples_per_temperature"]),
        "hidden": cohort_n * len(config["diagnostic_conditions"]),
        "masking": cohort_n * len(config["diagnostic_conditions"]) * len(config["source_masking"]["conditions"]),
    }
    models = [args.model_key] if args.model_key else list(config["models"])
    summaries = {}
    for model_key in models:
        root = output_root / "models" / model_key
        paths = {
            "candidate": root / "checkpoints" / "03_candidate_preference_standardized.jsonl",
            "sampling": root / "checkpoints" / "04_boundary_sampling_standardized.jsonl",
            "hidden": root / "checkpoints" / "05_hidden_state_capture_standardized.jsonl",
            "masking": root / "checkpoints" / "06_source_masking_standardized.jsonl",
        }
        counts = {name: count_jsonl(path) for name, path in paths.items()}
        hidden_rows = read_jsonl(paths["hidden"]) if paths["hidden"].exists() else []
        hidden_npz_ok = 0
        for row in hidden_rows:
            npz_path = output_root / row["npz_file"]
            if npz_path.exists() and sha256_file(npz_path) == row["npz_sha256"]:
                arr = np.load(npz_path)
                if tuple(arr["hidden"].shape) == tuple(row["hidden_shape"]):
                    hidden_npz_ok += 1
        complete = counts == expected and hidden_npz_ok == expected["hidden"]
        summaries[model_key] = {
            "expected": expected,
            "counts": counts,
            "hidden_npz_ok": hidden_npz_ok,
            "complete": complete,
        }
        atomic_write_json(root / "manifests" / "diagnostic_quality_gate_summary.json", summaries[model_key])
        if not complete:
            print(f"[WARN] {model_key} incomplete: {summaries[model_key]}")
        else:
            print(f"[OK] {model_key} complete")
    atomic_write_json(output_root / "diagnostic_quality_gate_summary.json", {"created_at_utc": utc_now(), "cohort_size": cohort_n, "models": summaries})


if __name__ == "__main__":
    main()
