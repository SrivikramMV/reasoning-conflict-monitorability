

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from analysis import run_automatic_analysis
from io_utils import read_jsonl, sha256_file, utc_timestamp, write_json
from study import FinalGemmaStudy


MODEL_STAGES = [
    "clean",
    "main",
    "candidate",
    "sampling",
    "visibility",
    "transparency",
    "masking",
]
ALL_STAGES = MODEL_STAGES + ["analysis", "archive"]


def verify_bundle(bundle_root: Path) -> dict:
    manifest_path = bundle_root / "bundle_checksums.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = bundle_root / relative
        if not path.exists():
            failures.append(f"missing: {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise RuntimeError("Bundle verification failed:\n" + "\n".join(failures))
    return manifest


def run_quality_gate(study: FinalGemmaStudy) -> dict:


    paths = study.paths
    config = study.config
    hard_failures: list[str] = []
    review_warnings: list[str] = []
    statistics: dict[str, object] = {}

    def load(name: str) -> list[dict]:
        return read_jsonl(paths.drive_checkpoints / f"{name}.jsonl")

    def check_exact(
        stage: str,
        rows: list[dict],
        key: str,
        expected: set[str],
    ) -> dict[str, dict]:
        values = [str(row[key]) for row in rows]
        counts = {value: values.count(value) for value in set(values)}
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        observed = set(values)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if duplicates:
            hard_failures.append(f"{stage}: duplicate {key} values: {duplicates[:10]}")
        if missing:
            hard_failures.append(f"{stage}: missing {len(missing)} expected rows: {missing[:10]}")
        if extra:
            hard_failures.append(f"{stage}: found {len(extra)} unexpected rows: {extra[:10]}")
        statistics[stage] = {
            "rows": len(rows),
            "expected_rows": len(expected),
            "missing_rows": len(missing),
            "extra_rows": len(extra),
        }
        return {str(row[key]): row for row in rows}

    pair_ids = [pair["pair_id"] for pair in study.pairs]
    clean = load("01_clean_controls")
    expected_clean = {
        f"{pair_id}::{role}"
        for pair_id in pair_ids
        for role in ("original", "counterfactual")
    }
    check_exact("clean", clean, "prompt_id", expected_clean)

    main = load("02_main_interventions")
    condition_names = [condition["name"] for condition in config["main_interventions"]]
    expected_main = {
        f"{pair_id}::{condition}"
        for pair_id in pair_ids
        for condition in condition_names
    }
    check_exact("main", main, "intervention_id", expected_main)

    clean_truncations = sum(bool(row.get("truncated")) for row in clean)
    main_truncations = sum(bool(row.get("truncated")) for row in main)
    clean_parse_failures = sum(row.get("parse_status") != "ok" for row in clean)
    main_parse_failures = sum(row.get("parse_status") != "ok" for row in main)
    statistics["generation_quality"] = {
        "clean_truncations": clean_truncations,
        "main_truncations": main_truncations,
        "clean_parse_failures": clean_parse_failures,
        "main_parse_failures": main_parse_failures,
        "clean_final_accuracy": (
            sum(bool(row.get("final_correct")) for row in clean) / len(clean) if clean else None
        ),
        "eligible_main_rows": sum(bool(row.get("clean_source_eligible")) for row in main),
    }
    if clean_truncations or main_truncations:
        review_warnings.append(
            f"Generation caps were reached in {clean_truncations} clean and "
            f"{main_truncations} main rows."
        )
    if clean_parse_failures or main_parse_failures:
        review_warnings.append(
            f"Response parsing was unresolved in {clean_parse_failures} clean and "
            f"{main_parse_failures} main rows."
        )

    candidate_config = config["candidate_preference"]
    eligible_candidate = [
        row
        for row in main
        if row.get("condition") == candidate_config["primary_condition"]
        and row.get("category") in candidate_config["eligible_categories"]
        and row.get("clean_source_eligible")
        and row.get("final_branch") in {"Q", "QSTAR"}
        and row.get("parse_status") == "ok"
    ]
    candidate = load("03_candidate_preference")
    candidate_ids = [str(row["candidate_id"]) for row in candidate]
    if len(candidate_ids) != len(set(candidate_ids)):
        hard_failures.append("candidate: duplicate candidate_id values")
    expected_primary_candidates = {
        f"{row['intervention_id']}::{candidate_config['primary_template']}"
        for row in eligible_candidate
    }
    observed_primary_candidates = {
        str(row["candidate_id"])
        for row in candidate
        if row.get("template") == candidate_config["primary_template"]
    }
    missing_primary = sorted(expected_primary_candidates - observed_primary_candidates)
    if missing_primary:
        hard_failures.append(
            f"candidate: missing {len(missing_primary)} primary candidate scores: "
            f"{missing_primary[:10]}"
        )
    statistics["candidate"] = {
        "rows": len(candidate),
        "expected_primary_rows": len(expected_primary_candidates),
        "observed_primary_rows": len(observed_primary_candidates),
    }

    primary_candidate_rows = [
        row
        for row in candidate
        if row.get("template") == candidate_config["primary_template"]
    ]
    sampling_config = config["sampling_stability"]
    expected_samples = {
        f"{row['intervention_id']}::t{temperature}::s{sample_index:02d}"
        for row in primary_candidate_rows
        for temperature in sampling_config["temperatures"]
        for sample_index in range(int(sampling_config["samples_per_temperature"]))
    }
    samples = load("04_sampling_stability")
    check_exact("sampling", samples, "sample_id", expected_samples)

    visibility_config = config["visibility_latency"]
    eligible_visibility = [
        row
        for row in main
        if row.get("category") in visibility_config["eligible_categories"]
        and row.get("condition") in visibility_config["conditions"]
        and row.get("clean_source_eligible")
        and row.get("final_branch") in {"Q", "QSTAR"}
        and row.get("parse_status") == "ok"
        and row.get("final_answer_token_ids")
    ]
    visibility = load("05_visibility_latency")
    visibility_by_id = check_exact(
        "visibility",
        visibility,
        "intervention_id",
        {str(row["intervention_id"]) for row in eligible_visibility},
    )
    feature_failures = []
    for intervention_id, row in visibility_by_id.items():
        feature_path = paths.feature_dir / row["feature_file"]
        if not feature_path.exists():
            feature_failures.append(f"missing {intervention_id}")
        elif sha256_file(feature_path) != row.get("feature_sha256"):
            feature_failures.append(f"hash mismatch {intervention_id}")
    if feature_failures:
        hard_failures.append(
            f"visibility features: {len(feature_failures)} failures: {feature_failures[:10]}"
        )

    transparency_config = config["transparency"]
    transparency_conditions = (
        ("conflict", "matched")
        if transparency_config["include_matched_controls"]
        else ("conflict",)
    )
    expected_transparency = {
        f"{pair_id}::{instruction}::{condition}"
        for pair_id in pair_ids
        for instruction in transparency_config["instruction_variants"]
        for condition in transparency_conditions
    }
    transparency = load("06_transparency_audit")
    check_exact(
        "transparency",
        transparency,
        "transparency_id",
        expected_transparency,
    )

    masking_config = config["source_masking"]
    eligible_masking = [
        row
        for row in main
        if row.get("category") in masking_config["eligible_categories"]
        and row.get("condition") in masking_config["conditions"]
        and row.get("clean_source_eligible")
        and row.get("final_branch") in {"Q", "QSTAR"}
        and row.get("parse_status") == "ok"
    ]
    expected_masks = {
        f"{row['intervention_id']}::{mask}"
        for row in eligible_masking
        for mask in masking_config["masks"]
    }
    masks = load("07_source_masking")
    check_exact("source_masking", masks, "mask_id", expected_masks)
    none_rows = [row for row in masks if row.get("mask") == "none"]
    none_mismatches = [
        row["mask_id"]
        for row in none_rows
        if row.get("masked_branch") != row.get("natural_branch")
    ]
    statistics["source_masking_controls"] = {
        "unmasked_rows": len(none_rows),
        "unmasked_branch_mismatches": len(none_mismatches),
    }
    if none_mismatches:
        review_warnings.append(
            f"The eager-attention unmasked control changed branch in "
            f"{len(none_mismatches)} cases: {none_mismatches[:10]}"
        )

    for stage in [
        "01_clean_controls",
        "02_main_interventions",
        "03_candidate_preference",
        "04_sampling_stability",
        "05_visibility_latency",
        "06_transparency_audit",
        "07_source_masking",
    ]:
        if not (paths.manifest_dir / f"{stage}.complete.json").exists():
            hard_failures.append(f"Missing completion manifest for {stage}")

    expected_review_rows = {
        "main_taxonomy_review.jsonl": len(main),
        "transparency_trace_grounding_review.jsonl": len(transparency),
    }
    for review_file in [
        paths.drive_root / "review_packets" / "main_taxonomy_review.jsonl",
        paths.drive_root / "review_packets" / "transparency_trace_grounding_review.jsonl",
    ]:
        if not review_file.exists():
            hard_failures.append(f"Missing manual review packet: {review_file.name}")
            continue
        review_count = len(read_jsonl(review_file))
        if review_count != expected_review_rows[review_file.name]:
            hard_failures.append(
                f"Manual review packet {review_file.name} has {review_count} rows; "
                f"expected {expected_review_rows[review_file.name]}."
            )

    report = {
        "created_at_utc": utc_timestamp(),
        "status": "PASSED" if not hard_failures else "FAILED",
        "ready_for_manual_adjudication": not hard_failures,
        "hard_failures": hard_failures,
        "review_warnings": review_warnings,
        "statistics": statistics,
    }
    write_json(paths.drive_root / "QUALITY_GATE.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if hard_failures:
        raise RuntimeError(
            "Final quality gate failed. Check QUALITY_GATE.json; all checkpoints were preserved."
        )
    return report

def make_archive(drive_root: Path) -> Path:
    base = drive_root.parent / (drive_root.name + "_results")
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=drive_root))
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--pair-ids", nargs="*")
    parser.add_argument("--stages", nargs="*", choices=ALL_STAGES + ["all"], default=["all"])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        original = sys.argv
        sys.argv = [original[0], *argv]
        try:
            args = parse_args()
        finally:
            sys.argv = original
    else:
        args = parse_args()
    bundle_root = args.bundle_root.resolve()
    manifest = verify_bundle(bundle_root)
    print(
        f"Verified final study bundle {manifest['bundle_version']} "
        f"({len(manifest['files'])} files).",
        flush=True,
    )
    selected = ALL_STAGES if "all" in args.stages else args.stages
    if args.validate_only:
        selected = []
    load_model = any(stage in MODEL_STAGES for stage in selected)
    study = FinalGemmaStudy(
        bundle_root=bundle_root,
        local_root=args.local_root,
        drive_root=args.drive_root,
        hf_token=args.hf_token,
        pair_ids=args.pair_ids,
        load_model_now=load_model,
    )
    stage_functions = {
        "clean": study.run_clean_controls,
        "main": study.run_main_interventions,
        "candidate": study.run_candidate_preference,
        "sampling": study.run_sampling_stability,
        "visibility": study.run_visibility_latency,
        "transparency": study.run_transparency_audit,
        "masking": study.run_source_masking,
    }
    for stage in MODEL_STAGES:
        if stage in selected:
            print(f"\n=== Running stage: {stage} ===\n", flush=True)
            stage_functions[stage]()

    quality_report = None
    if set(MODEL_STAGES).issubset(selected):
        print("\n=== Running final completeness gate ===\n", flush=True)
        quality_report = run_quality_gate(study)

    if "analysis" in selected:
        print("\n=== Running automatic analysis ===\n", flush=True)
        summaries = run_automatic_analysis(study.paths)
        print(json.dumps(summaries, indent=2), flush=True)

    archive = None
    if "archive" in selected:
        print("\n=== Packaging Drive results ===\n", flush=True)
        archive = make_archive(study.paths.drive_root)
        print(f"Result archive: {archive}", flush=True)

    completion = {
        "study_id": study.config["study_id"],
        "completed_at_utc": utc_timestamp(),
        "selected_stages": selected,
        "selected_pair_count": len(study.pairs),
        "result_archive": None if archive is None else str(archive),
        "quality_gate_status": None if quality_report is None else quality_report["status"],
        "manual_review_still_required": [
            "main correction-aware taxonomy",
            "trace-grounding of claimed transparent corrections",
        ],
    }
    write_json(study.paths.drive_root / "RUN_COMPLETE.json", completion)
    print("\nThe requested stages completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
