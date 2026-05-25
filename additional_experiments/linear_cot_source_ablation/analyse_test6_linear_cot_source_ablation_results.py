





from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "Results" / "test6_linear_cot_source_ablation_gemma4_e2b"
RESULT_JSONL = RESULT_DIR / "test6_linear_cot_source_ablation_gemma4_e2b.jsonl"
ANALYSIS_DIR = RESULT_DIR / "analysis"
LABEL_CSV = ANALYSIS_DIR / "test6_linear_cot_source_ablation_labels.csv"
SUMMARY_MD = ANALYSIS_DIR / "test6_linear_cot_source_ablation_summary.md"


BRIDGE_TERMS = (
    "scratchpad",
    "original prompt",
    "re-check",
    "recheck",
    "compare",
    "comparison",
    "discrepancy",
    "self-correction",
    "mistake",
    "wait",
)


def normalize_math(text: str) -> str:
    text = re.sub(
        r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}",
        r"\1/\2",
        text,
    )
    text = text.lower().replace(chr(8722), "-").replace(chr(8211), "-")
    for token in (
        "\\mathbf",
        "\\text",
        "\\left",
        "\\right",
        "\\quad",
        "\\implies",
        "\\times",
        "\\beginarray",
        "\\endarray",
        "\\hline",
    ):
        text = text.replace(token, "")
    for char in "$`*{}[]()\\":
        text = text.replace(char, "")
    text = re.sub(r"\s+", "", text)
    while "+-" in text:
        text = text.replace("+-", "-")
    while "--" in text:
        text = text.replace("--", "+")
    for var in "xyz":
        text = re.sub(rf"(?<![0-9])1{var}", var, text)
        text = text.replace(f"-1{var}", f"-{var}")
        text = text.replace(f"+1{var}", f"+{var}")
    return text


def equation_lines(question: str) -> list[str]:
    return [line.strip() for line in question.splitlines() if "=" in line]


def changed_pairs(row: dict) -> list[tuple[str, str]]:
    original = equation_lines(row["original_question"])
    counterfactual = equation_lines(row["counterfactual_question"])
    changed = []
    for q_line, qstar_line in zip(original, counterfactual):
        if normalize_math(q_line) != normalize_math(qstar_line):
            changed.append((q_line, qstar_line))
    return changed or list(zip(original, counterfactual))


def state_positions(row: dict) -> list[tuple[int, str]]:
    text = row.get("final_answer") or ""
    normalized = normalize_math(text)
    out: list[tuple[int, str]] = []
    for q_line, qstar_line in changed_pairs(row):
        for state, pattern in (("Q", q_line), ("Q*", qstar_line)):
            normalized_pattern = normalize_math(pattern)
            start = 0
            while True:
                idx = normalized.find(normalized_pattern, start)
                if idx < 0:
                    break
                out.append((idx, state))
                start = idx + 1
    return sorted(out)


def boundary_states(row: dict) -> tuple[str, str, int, int]:
    positions = state_positions(row)
    counts = Counter(state for _, state in positions)
    if not positions:
        return "none", "none", counts["Q"], counts["Q*"]
    return positions[0][1], positions[-1][1], counts["Q"], counts["Q*"]


def final_label(row: dict) -> str:
    if row.get("status") != "ok":
        return "RUNTIME_ERROR"

    first, terminal, q_count, qstar_count = boundary_states(row)
    if qstar_count > 0 and terminal == "Q":
        return "MIXED_QSTAR_THEN_PROMPT_Q"
    if terminal == "Q*":
        return "FOLLOW_QSTAR"
    if terminal == "Q":
        return "PROMPT_Q_RECOMPUTE"
    return "NO_BOUNDARY_STATE_DETECTED"


def compact(text: str, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def derived_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        first, terminal, q_count, qstar_count = boundary_states(row)
        final = row.get("final_answer") or ""
        out.append(
            {
                "intervention_id": row["intervention_id"],
                "base_item_id": row["base_item_id"],
                "probe_variant": row["probe_variant"],
                "probe_family": row["probe_family"],
                "prior_full_cot_label": row["prior_full_cot_label"],
                "status": row.get("status"),
                "error_type": row.get("error_type", ""),
                "error_message": row.get("error_message", ""),
                "first_boundary_state": first,
                "terminal_boundary_state": terminal,
                "q_changed_equation_mentions": q_count,
                "qstar_changed_equation_mentions": qstar_count,
                "visible_bridge_terms": any(term in final.lower() for term in BRIDGE_TERMS),
                "analysis_label": final_label(row),
                "final_answer_start": compact(final, 320),
                "final_answer_tail": compact(final[-900:], 320),
            }
        )
    return out


def write_csv(records: list[dict]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with LABEL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_summary(records: list[dict]) -> None:
    status_counts = Counter(record["status"] for record in records)
    label_counts = Counter(record["analysis_label"] for record in records)
    by_prior_variant: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_item: dict[str, Counter] = defaultdict(Counter)
    by_variant: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        by_prior_variant[(record["prior_full_cot_label"], record["probe_variant"])][record["analysis_label"]] += 1
        by_item[record["base_item_id"]][record["analysis_label"]] += 1
        by_variant[record["probe_variant"]][record["analysis_label"]] += 1

    lines = [
        "# Test 6 Linear CoT Source-Ablation Derived Summary",
        "",
        f"Raw result file read: `{RESULT_JSONL}`",
        f"Derived label CSV written: `{LABEL_CSV}`",
        "",
        f"Status counts: {dict(status_counts)}",
        f"Label counts: {dict(label_counts)}",
        "",
        "## By Prior Label And Variant",
        "",
    ]
    for key, counts in sorted(by_prior_variant.items()):
        rendered = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        lines.append(f"- `{key[0]}` / `{key[1]}`: {rendered}")

    lines.extend(["", "## By Item", ""])
    for item, counts in sorted(by_item.items()):
        rendered = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        lines.append(f"- `{item}`: {rendered}")

    lines.extend(["", "## By Variant", ""])
    for variant, counts in sorted(by_variant.items()):
        rendered = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered}")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [
        json.loads(line)
        for line in RESULT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = derived_rows(rows)
    write_csv(records)
    write_summary(records)
    print(f"Read {len(rows)} rows.")
    print(f"Wrote {LABEL_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
