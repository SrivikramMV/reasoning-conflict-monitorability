import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_B_DIR = (
    ROOT
    / "Results"
    / "Test 4 - 300 questions all CoT"
    / "Stage B results"
    / "Gemma_CoT_Stage_B"
)
MANUAL_DIR = STAGE_B_DIR / "analysis" / "manual_review"
PACKETS = MANUAL_DIR / "test4_stage_b_review_packets.jsonl"
SHARDS = [
    MANUAL_DIR / "judged_shard_000_199.jsonl",
    MANUAL_DIR / "judged_shard_200_399.jsonl",
    MANUAL_DIR / "judged_shard_400_599.jsonl",
]
OUT_JSON = MANUAL_DIR / "test4_stage_b_MANUAL_JUDGED.pretty.json"
OUT_CSV = MANUAL_DIR / "test4_stage_b_MANUAL_JUDGED.csv"
CHECK_REPORT = MANUAL_DIR / "test4_stage_b_manual_label_checks.md"


VALID_LABELS = {
    "FOLLOW",
    "COT_STAGE_CORRECTION",
    "FINAL_STAGE_TRANSPARENT_CORRECTION",
    "SILENT_BYPASS_REANCHOR",
    "SILENT_BYPASS_RECOMPUTE",
    "ANSWER_ONLY_FOLLOW",
    "ANSWER_ONLY_BYPASS",
    "STRUCTURAL_ARTIFACT",
    "DATASET_ISSUE_EXCLUDE",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    packets = load_jsonl(PACKETS)
    packet_by_id = {p["intervention_id"]: p for p in packets}
    labels = []
    missing_shards = [str(path) for path in SHARDS if not path.exists()]
    if missing_shards:
        raise FileNotFoundError("Missing judged shard(s):\n" + "\n".join(missing_shards))

    for shard in SHARDS:
        labels.extend(load_jsonl(shard))

    seen = Counter(row["intervention_id"] for row in labels)
    duplicates = [iid for iid, count in seen.items() if count > 1]
    missing = [p["intervention_id"] for p in packets if p["intervention_id"] not in seen]
    unknown = [row["intervention_id"] for row in labels if row["intervention_id"] not in packet_by_id]

    merged = []
    issues = []
    for row in labels:
        iid = row["intervention_id"]
        packet = packet_by_id.get(iid, {})
        merged_row = {
            **{k: packet.get(k) for k in [
                "intervention_id", "base_item_id", "study_block", "intervention_family",
                "intervention_type", "injection_mode", "answer_format", "category",
                "difficulty", "target_factor", "edit_type", "original_answer",
                "counterfactual_answer", "expected_wrong_answer", "generated_token_count",
                "max_new_tokens",
            ]},
            **row,
        }
        merged.append(merged_row)

        label = row.get("manual_label")
        if label not in VALID_LABELS:
            issues.append((iid, "invalid_label", str(label)))
        if label == "FINAL_STAGE_TRANSPARENT_CORRECTION" and row.get("visible_bridge_type") in {"", None, "none"}:
            issues.append((iid, "transparent_without_bridge", row.get("judge_notes", "")))
        if label in {"SILENT_BYPASS_REANCHOR", "SILENT_BYPASS_RECOMPUTE"} and row.get("visible_bridge_type") not in {"", None, "none"}:
            issues.append((iid, "silent_with_bridge_marked", row.get("visible_bridge_type", "")))
        if label == "COT_STAGE_CORRECTION" and row.get("cot_boundary_state") not in {"original_correct", "already_corrected"}:
            issues.append((iid, "cot_correction_boundary_state_check", row.get("cot_boundary_state", "")))
        if label in {"ANSWER_ONLY_FOLLOW", "ANSWER_ONLY_BYPASS"} and packet.get("answer_format") != "answer_only":
            issues.append((iid, "answer_only_label_on_normal_row", packet.get("answer_format", "")))

    merged.sort(key=lambda r: [p["intervention_id"] for p in packets].index(r["intervention_id"]))

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    write_csv(OUT_CSV, merged)

    counts = Counter(row["manual_label"] for row in merged)
    by_type = defaultdict(Counter)
    by_cat = defaultdict(Counter)
    by_target = defaultdict(Counter)
    by_conf = Counter(row.get("confidence", "") for row in merged)
    for row in merged:
        by_type[row.get("intervention_type", "")][row["manual_label"]] += 1
        by_cat[row.get("category", "")][row["manual_label"]] += 1
        by_target[row.get("target_factor", "")][row["manual_label"]] += 1

    def table(counter):
        lines = ["| Label | Count |", "|---|---:|"]
        for key, value in counter.most_common():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)

    def group_table(groups):
        labels_all = sorted({label for counts in groups.values() for label in counts})
        lines = ["| Group | Total | " + " | ".join(labels_all) + " |"]
        lines.append("|---|---:|" + "|".join(["---:"] * len(labels_all)) + "|")
        for group in sorted(groups):
            total = sum(groups[group].values())
            vals = [str(groups[group].get(label, 0)) for label in labels_all]
            lines.append(f"| {group} | {total} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = []
    report.append("# Manual Label Consistency Checks\n")
    report.append(f"Total packets: {len(packets)}")
    report.append(f"Total labels: {len(labels)}")
    report.append(f"Duplicates: {len(duplicates)}")
    report.append(f"Missing: {len(missing)}")
    report.append(f"Unknown: {len(unknown)}")
    report.append(f"Consistency issues: {len(issues)}\n")
    report.append("## Label Counts\n")
    report.append(table(counts))
    report.append("\n## Confidence Counts\n")
    report.append(table(by_conf))
    report.append("\n## By Intervention Type\n")
    report.append(group_table(by_type))
    report.append("\n## By Category\n")
    report.append(group_table(by_cat))
    report.append("\n## By Target Factor\n")
    report.append(group_table(by_target))
    report.append("\n## Issues\n")
    if issues:
        report.append("| Intervention | Issue | Detail |")
        report.append("|---|---|---|")
        for iid, kind, detail in issues:
            detail = str(detail).replace("\n", " ")[:500]
            report.append(f"| `{iid}` | {kind} | {detail} |")
    else:
        report.append("No consistency issues found.")
    CHECK_REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {CHECK_REPORT}")
    print(counts)
    if issues:
        print(f"Issues: {len(issues)}")


if __name__ == "__main__":
    main()
