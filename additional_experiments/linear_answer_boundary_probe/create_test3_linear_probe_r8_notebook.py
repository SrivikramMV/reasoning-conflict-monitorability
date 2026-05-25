import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "gemma4_e2b_stage_b_colab_R7.ipynb"
OUT_NOTEBOOK = ROOT / "gemma4_e2b_linear_answer_boundary_probe_R8.ipynb"


def source(text):
    return [line + "\n" for line in text.rstrip("\n").split("\n")]


def main():
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = source(
        """# Gemma 4 E2B Linear Answer-Boundary Probe Runner R8

This notebook runs the **focused Test 3 linear answer-boundary probe** dataset.

It loads:

```text
test3_focused_linear_answer_boundary_probe_dataset.jsonl
```

For each probe row it:

1. Builds the original linear-system prompt `Q`.
2. Prefills Gemma with a full counterfactual `Q*` CoT plus the probe-specific boundary manipulation.
3. Lets Gemma generate the final answer.
4. Saves the raw generation, parsed CoT/final answer, token IDs, and all probe metadata.
5. Checkpoints every row and syncs results to Google Drive.

Use a Colab GPU runtime and run top-to-bottom."""
    )

    notebook["cells"][11]["source"] = source(
        """## 6. Upload / Load Linear Boundary Probe Dataset

Upload this file when prompted:

```text
test3_focused_linear_answer_boundary_probe_dataset.jsonl
```"""
    )

    notebook["cells"][12]["source"] = source(
        '''INTERVENTIONS_PATH = Path("/content/test3_focused_linear_answer_boundary_probe_dataset.jsonl")


def maybe_upload_interventions(path=INTERVENTIONS_PATH):
    path = Path(path)
    if path.exists():
        print(f"Found probe dataset: {path}")
        return path

    from google.colab import files
    print("Upload test3_focused_linear_answer_boundary_probe_dataset.jsonl")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file was uploaded.")

    uploaded_name = next(iter(uploaded.keys()))
    uploaded_path = Path("/content") / uploaded_name
    if uploaded_path != path:
        path.write_bytes(uploaded_path.read_bytes())
        print(f"Copied uploaded file to {path}")
    return path


def load_interventions(path=INTERVENTIONS_PATH):
    path = maybe_upload_interventions(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_intervention_line"] = line_number
            rows.append(row)

    ids = [row["intervention_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Probe file contains duplicate intervention_id values.")

    print(f"Loaded {len(rows)} linear answer-boundary probe rows from {path}")
    return rows


interventions = load_interventions(INTERVENTIONS_PATH)
interventions[:2]'''
    )

    notebook["cells"][13]["source"] = source(
        """## 7. Output / Save Utilities

The runner writes one JSONL record immediately after each probe row and syncs to Drive every few records."""
    )

    notebook["cells"][14]["source"] = source(
        '''LOCAL_OUTPUT_DIR = Path("/content/gemma_linear_boundary_probe_outputs")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/Gemma_CoT_Test3_Linear_Boundary_Probe")
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME = "test3_linear_answer_boundary_probe_gemma4_e2b"
RUN_JSONL = LOCAL_OUTPUT_DIR / f"{RUN_NAME}.jsonl"
DRIVE_JSONL = DRIVE_OUTPUT_DIR / f"{RUN_NAME}.jsonl"


def load_existing_records(*paths):
    records = []
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                intervention_id = record.get("intervention_id")
                if intervention_id and intervention_id not in seen:
                    records.append(record)
                    seen.add(intervention_id)
    return records


def append_jsonl_record(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        f.flush()
        os.fsync(f.fileno())


def save_final_artifacts(records, output_dir=LOCAL_OUTPUT_DIR, run_name=RUN_NAME):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_name}.jsonl"
    pretty_path = output_dir / f"{run_name}.pretty.json"
    csv_path = output_dir / f"{run_name}.csv"
    zip_path = output_dir / f"{run_name}.zip"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")

    with open(pretty_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    all_keys = []
    for record in records:
        for key in record.keys():
            if key not in all_keys:
                all_keys.append(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for record in records:
            row = {}
            for key in all_keys:
                value = record.get(key, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                row[key] = value
            writer.writerow(row)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [jsonl_path, pretty_path, csv_path]:
            zf.write(path, arcname=path.name)

    return {
        "jsonl": str(jsonl_path),
        "pretty_json": str(pretty_path),
        "csv": str(csv_path),
        "zip": str(zip_path),
    }


def copy_artifacts_to_drive(local_paths, drive_output_dir=DRIVE_OUTPUT_DIR):
    drive_output_dir = Path(drive_output_dir)
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, path in local_paths.items():
        src = Path(path)
        dst = drive_output_dir / src.name
        shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


print(f"Local output dir: {LOCAL_OUTPUT_DIR}")
print(f"Drive output dir: {DRIVE_OUTPUT_DIR}")'''
    )

    notebook["cells"][15]["source"] = source(
        """## 8. Run Linear Boundary Probe Batch

For the real run, keep `LIMIT = None`.

Optional filters are available for smoke tests or targeted subsets."""
    )

    notebook["cells"][16]["source"] = source(
        '''# Batch config
LIMIT = None                    # None for all probe rows, or an integer for a smoke test.
START_INDEX = 0                 # 0-based inclusive index after filtering.
END_INDEX = None                # None or 0-based exclusive index after filtering.
RUN_ONLY_PROBE_FAMILIES = None  # Example: {"last_frame_probe"}; None means all.
RUN_ONLY_VARIANTS = None        # Example: {"frame_qstar_self_check"}; None means all.
RUN_ONLY_CATEGORIES = None      # Example: {"linear_system_3var"}; None means all.
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
SYNC_EVERY = 5
RESUME = True


def selected_interventions(all_rows):
    rows = all_rows
    if RUN_ONLY_PROBE_FAMILIES is not None:
        rows = [r for r in rows if r.get("probe_family") in RUN_ONLY_PROBE_FAMILIES]
    if RUN_ONLY_VARIANTS is not None:
        rows = [r for r in rows if r.get("probe_variant") in RUN_ONLY_VARIANTS]
    if RUN_ONLY_CATEGORIES is not None:
        rows = [r for r in rows if r.get("category") in RUN_ONLY_CATEGORIES]
    rows = rows[START_INDEX:END_INDEX]
    if isinstance(LIMIT, int):
        rows = rows[:LIMIT]
    return rows


def make_stage_b_record(intervention, generation, elapsed_seconds):
    record = dict(intervention)
    record.update({
        "model_id": MODEL_ID,
        "created_at_utc": utc_now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "do_sample_run": DO_SAMPLE,
        "temperature_run": TEMPERATURE if DO_SAMPLE else None,
        "top_p_run": TOP_P if DO_SAMPLE else None,
        "top_k_run": TOP_K if DO_SAMPLE else None,
        "cot": generation["cot"],
        "final_answer": generation["final_answer"],
        "raw_generation": generation["raw_generation"],
        "generated_continuation": generation["generated_continuation"],
        "generated_token_ids": generation["generated_token_ids"],
        "generated_token_count": generation["generated_token_count"],
        "parsed": generation["parsed"],
    })
    return record


def run_stage_b_batch(all_interventions):
    run_rows = selected_interventions(all_interventions)
    existing_records = load_existing_records(DRIVE_JSONL, RUN_JSONL) if RESUME else []
    existing_by_id = {r["intervention_id"]: r for r in existing_records if "intervention_id" in r}

    if existing_records and not RUN_JSONL.exists():
        with open(RUN_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")

    total = len(run_rows)
    new_count = 0
    print(f"Probe rows selected: {total}")
    print(f"Existing records found: {len(existing_by_id)}")
    print(f"Writing incremental JSONL to: {RUN_JSONL}")

    for index, intervention in enumerate(run_rows, start=1):
        intervention_id = intervention["intervention_id"]
        if intervention_id in existing_by_id:
            print(f"[{index}/{total}] SKIP existing {intervention_id}")
            continue

        question = intervention["question"]
        max_new_tokens = int(intervention.get("max_new_tokens") or 2048)
        print(f"[{index}/{total}] Running {intervention_id}")
        print(
            "  "
            f"family={intervention.get('probe_family')} "
            f"variant={intervention.get('probe_variant')} "
            f"category={intervention.get('category')} "
            f"prior={intervention.get('prior_full_cot_label')} "
            f"max_new_tokens={max_new_tokens}"
        )
        print("  Q:", question[:180].replace("\\n", " | "))

        t0 = time.time()
        try:
            generation = generate_from_injected_prefix(
                question=question,
                injected_raw_prefix=intervention["injected_raw_prefix"],
                max_new_tokens=max_new_tokens,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )
            elapsed = time.time() - t0
            record = make_stage_b_record(intervention, generation, elapsed)
            record["status"] = "ok"
        except Exception as exc:
            elapsed = time.time() - t0
            record = dict(intervention)
            record.update({
                "model_id": MODEL_ID,
                "created_at_utc": utc_now_iso(),
                "elapsed_seconds": round(elapsed, 3),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(f"ERROR on {intervention_id}: {type(exc).__name__}: {exc}")

        append_jsonl_record(RUN_JSONL, record)
        existing_by_id[intervention_id] = record
        new_count += 1

        preview = record.get("final_answer", "")
        if preview:
            print("  Final preview:", repr(preview[:220]))
        print(f"  Elapsed: {elapsed:.1f}s, new records this run: {new_count}")

        if new_count % SYNC_EVERY == 0:
            shutil.copy2(RUN_JSONL, DRIVE_JSONL)
            print(f"  Synced partial JSONL to Drive: {DRIVE_JSONL}")

    shutil.copy2(RUN_JSONL, DRIVE_JSONL)
    print(f"Final incremental JSONL synced to Drive: {DRIVE_JSONL}")

    selected_ids = {r["intervention_id"] for r in run_rows}
    final_records = [r for r in load_existing_records(RUN_JSONL) if r.get("intervention_id") in selected_ids]
    print(f"Final record count for selected probe rows: {len(final_records)}")
    return final_records


stage_b_results = run_stage_b_batch(interventions)'''
    )

    notebook["cells"][21]["source"] = source("## 11. Quick Summary Check")
    notebook["cells"][22]["source"] = source(
        '''from collections import Counter

records = load_existing_records(RUN_JSONL)
print(f"Records: {len(records)}")
print("Status:", Counter(r.get("status") for r in records))
print("Probe family:", Counter(r.get("probe_family") for r in records))
print("Probe variant:", Counter(r.get("probe_variant") for r in records))
print("Category:", Counter(r.get("category") for r in records))
print("Prior Test 4 label:", Counter(r.get("prior_full_cot_label") for r in records))
print("Answer format:", Counter(r.get("answer_format") for r in records))'''
    )

    OUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_NOTEBOOK)


if __name__ == "__main__":
    main()
