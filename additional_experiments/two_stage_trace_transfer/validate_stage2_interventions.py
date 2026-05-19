import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Test 2 data"
INTERVENTIONS = DATA_DIR / "stage2_stage_b_interventions_600.jsonl"


REQUIRED_FIELDS = [
    "intervention_id",
    "base_item_id",
    "study_block",
    "intervention_family",
    "intervention_type",
    "injection_mode",
    "answer_format",
    "question",
    "original_question",
    "original_answer",
    "expected_wrong_answer",
    "answer_type",
    "category",
    "difficulty",
    "target_factor",
    "source_prompt_id",
    "injected_thought",
    "injected_raw_prefix",
    "append_answer_channel",
    "max_new_tokens",
]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_jsonl(INTERVENTIONS)
    assert len(rows) == 600, f"Expected 600 interventions, got {len(rows)}"
    ids = [row["intervention_id"] for row in rows]
    assert len(ids) == len(set(ids)), "Duplicate intervention ids"

    for row in rows:
        for field in REQUIRED_FIELDS:
            assert field in row, f"{row.get('intervention_id')} missing {field}"
            assert row[field] not in ("", None, []), f"{row.get('intervention_id')} empty {field}"
        assert row["injected_raw_prefix"].startswith("<|channel>thought\n"), row["intervention_id"]
        if row["append_answer_channel"]:
            assert row["injected_raw_prefix"].endswith("<channel|>"), row["intervention_id"]
        if row["answer_format"] == "answer_only":
            assert "only give the final answer" in row["question"], row["intervention_id"]

    print("Stage B intervention validation passed.")
    print()
    print("Interventions:", len(rows))
    for key in ["intervention_family", "intervention_type", "answer_format", "category", "edit_type"]:
        print(key + ":")
        for name, count in Counter(row.get(key) for row in rows).most_common():
            print(f"  {name}: {count}")
        print()


if __name__ == "__main__":
    main()
