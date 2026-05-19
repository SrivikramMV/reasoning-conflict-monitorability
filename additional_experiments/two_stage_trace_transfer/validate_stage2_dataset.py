import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Test 2 data"
DATASET = DATA_DIR / "stage2_cot_faithfulness_dataset_180.jsonl"
STAGE_A_PROMPTS = DATA_DIR / "stage2_stage_a_prompts_300.jsonl"


REQUIRED_FIELDS = [
    "id",
    "study_block",
    "category",
    "difficulty",
    "target_factor",
    "intervention_intent",
    "original_question",
    "original_answer",
    "answer_type",
    "expected_solution_method",
    "method_stability",
    "original_solution_sketch",
    "stage_a_prompts",
]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    rows = load_jsonl(DATASET)
    prompts = load_jsonl(STAGE_A_PROMPTS)

    assert_true(len(rows) == 180, f"Expected 180 dataset rows, got {len(rows)}")
    assert_true(len(prompts) == 300, f"Expected 300 Stage A prompts, got {len(prompts)}")

    ids = [row["id"] for row in rows]
    assert_true(len(ids) == len(set(ids)), "Dataset ids are not unique")

    prompt_ids = [row["prompt_id"] for row in prompts]
    assert_true(len(prompt_ids) == len(set(prompt_ids)), "Stage A prompt ids are not unique")

    for row in rows:
        for field in REQUIRED_FIELDS:
            assert_true(field in row, f"{row.get('id')} missing field {field}")
            assert_true(row[field] not in ("", []), f"{row.get('id')} has empty field {field}")

        if row["study_block"] == "naturalistic_counterfactual_transfer":
            assert_true(row["counterfactual_question"], f"{row['id']} missing counterfactual question")
            assert_true(row["counterfactual_answer"], f"{row['id']} missing counterfactual answer")
            assert_true(row["original_answer"] != row["counterfactual_answer"], f"{row['id']} has same Q and Q* answer")
            assert_true("counterfactual" in row["stage_a_prompts"], f"{row['id']} missing counterfactual Stage A prompt")
        elif row["study_block"] == "edited_cot_ablation_seed":
            assert_true(row["counterfactual_question"] is None, f"{row['id']} edited seed should not have Q*")
            assert_true(row["edit_plan"], f"{row['id']} missing edit plan")
            assert_true(row["stage_a_prompts"] == ["original"], f"{row['id']} edited seed should only need original Stage A prompt")
        else:
            raise AssertionError(f"{row['id']} has unknown study block {row['study_block']}")

    prompt_base_ids = Counter(row["base_item_id"] for row in prompts)
    for row in rows:
        expected = 2 if row["study_block"] == "naturalistic_counterfactual_transfer" else 1
        assert_true(prompt_base_ids[row["id"]] == expected, f"{row['id']} has wrong number of Stage A prompts")

    print("Stage 2 dataset validation passed.")
    print()
    print("Dataset rows:", len(rows))
    print("Stage A prompts:", len(prompts))
    print()
    for name in ["study_block", "category", "difficulty", "target_factor"]:
        print(name + ":")
        for key, value in Counter(row[name] for row in rows).most_common():
            print(f"  {key}: {value}")
        print()

    print("Edited-CoT edit types:")
    edit_counts = Counter(row["edit_plan"]["edit_type"] for row in rows if row["study_block"] == "edited_cot_ablation_seed")
    for key, value in edit_counts.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
