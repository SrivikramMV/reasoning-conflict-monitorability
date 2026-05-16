import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pilot_question_pairs_50.jsonl"

REQUIRED_FIELDS = {
    "id",
    "category",
    "difficulty",
    "original_question",
    "counterfactual_question",
    "original_answer",
    "counterfactual_answer",
    "answer_type",
    "pair_change_description",
    "expected_solution_method",
    "method_stability",
    "original_solution_sketch",
    "counterfactual_solution_sketch",
}

EXPECTED_CATEGORY_COUNTS = {
    "arithmetic": 7,
    "algebra_1var": 8,
    "word_problem": 10,
    "linear_system_2var": 7,
    "linear_system_3var": 8,
    "probability_combinatorics": 5,
    "calculus_functions": 5,
}

EXPECTED_DIFFICULTY_COUNTS = {
    "easy": 10,
    "medium": 25,
    "hard": 15,
}


def load_jsonl(path):
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return examples


def main():
    examples = load_jsonl(DATA_PATH)

    if len(examples) != 50:
        raise AssertionError(f"Expected exactly 50 examples, found {len(examples)}")

    ids = [example.get("id") for example in examples]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise AssertionError(f"Duplicate ids found: {duplicate_ids}")

    for index, example in enumerate(examples, start=1):
        missing = REQUIRED_FIELDS - set(example)
        if missing:
            raise AssertionError(f"Example {index} ({example.get('id')}) missing fields: {sorted(missing)}")
        if str(example["original_answer"]) == str(example["counterfactual_answer"]):
            raise AssertionError(f"{example['id']} has identical original and counterfactual answers")

    category_counts = Counter(example["category"] for example in examples)
    difficulty_counts = Counter(example["difficulty"] for example in examples)

    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise AssertionError(
            f"Category distribution mismatch.\nExpected: {EXPECTED_CATEGORY_COUNTS}\nFound: {dict(category_counts)}"
        )

    if dict(difficulty_counts) != EXPECTED_DIFFICULTY_COUNTS:
        raise AssertionError(
            f"Difficulty distribution mismatch.\nExpected: {EXPECTED_DIFFICULTY_COUNTS}\nFound: {dict(difficulty_counts)}"
        )

    print("Validation passed.")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Examples: {len(examples)}")
    print("Category distribution:")
    for category, expected in EXPECTED_CATEGORY_COUNTS.items():
        print(f"  {category}: {category_counts[category]}/{expected}")
    print("Difficulty distribution:")
    for difficulty, expected in EXPECTED_DIFFICULTY_COUNTS.items():
        print(f"  {difficulty}: {difficulty_counts[difficulty]}/{expected}")


if __name__ == "__main__":
    main()
