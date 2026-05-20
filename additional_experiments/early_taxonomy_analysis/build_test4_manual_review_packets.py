import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_B_DIR = (
    ROOT
    / "Results"
    / "Test 4 - 300 questions all CoT"
    / "Stage B results"
    / "Gemma_CoT_Stage_B"
)
INPUT = STAGE_B_DIR / "stage2_stage_b_gemma4_e2b.jsonl"
OUT_DIR = STAGE_B_DIR / "analysis" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PACKETS_JSONL = OUT_DIR / "test4_stage_b_review_packets.jsonl"
PACKETS_PRETTY = OUT_DIR / "test4_stage_b_review_packets.pretty.json"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_generated_continuation(record):
    prefix = record.get("injected_raw_prefix") or ""
    continuation = record.get("generated_continuation") or ""

    if prefix.endswith("<channel|>"):
        return {
            "injected_cot": prefix,
            "generated_cot_continuation": "",
            "answer_channel_marker_source": "injected_prefix",
            "final_raw_from_continuation": continuation,
        }

    if "<channel|>" in continuation:
        before, after = continuation.split("<channel|>", 1)
        return {
            "injected_cot": prefix,
            "generated_cot_continuation": before,
            "answer_channel_marker_source": "generated_continuation",
            "final_raw_from_continuation": after,
        }

    return {
        "injected_cot": prefix,
        "generated_cot_continuation": continuation,
        "answer_channel_marker_source": "not_seen",
        "final_raw_from_continuation": "",
    }


def trim(text, limit=6000):
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n...[TRIMMED FOR PACKET; USE RAW FILE FOR FULL TEXT]...\n" + tail


def normalize(s):
    s = s or ""
    s = s.lower()
    s = s.replace("\\\\", "\\")
    s = re.sub(r"\s+", " ", s)
    return s


def answer_mentions(text, answer):
    text_n = normalize(text)
    parts = re.findall(r"-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?", str(answer or ""))
    if not parts:
        return False
    hits = 0
    for part in parts:
        p = part.replace(" ", "")
        if p in text_n.replace(" ", ""):
            hits += 1
    return hits >= max(1, len(parts))


def bridge_terms(text):
    terms = [
        "wait", "actually", "correction", "self-correction", "double check",
        "double-check", "recheck", "re-check", "re-read", "reread", "misread",
        "mistake", "earlier", "previous", "wrong", "instead", "but the prompt",
        "original problem", "problem statement", "provided in the prompt",
        "prompt stated", "does not match", "inconsistent",
    ]
    text_n = normalize(text)
    return [term for term in terms if term in text_n]


def first_changed_line(original_question, counterfactual_question):
    orig_lines = (original_question or "").splitlines()
    cf_lines = (counterfactual_question or "").splitlines()
    for i, (a, b) in enumerate(zip(orig_lines, cf_lines), start=1):
        if a != b:
            return {"line_number": i, "original": a, "counterfactual": b}
    return None


def build_packet(record):
    split = split_generated_continuation(record)
    final_answer = record.get("final_answer") or ""
    generated_cot = split["generated_cot_continuation"]
    injected = split["injected_cot"]

    packet = {
        "intervention_id": record["intervention_id"],
        "base_item_id": record.get("base_item_id"),
        "study_block": record.get("study_block"),
        "intervention_family": record.get("intervention_family"),
        "intervention_type": record.get("intervention_type"),
        "injection_mode": record.get("injection_mode"),
        "answer_format": record.get("answer_format"),
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "target_factor": record.get("target_factor"),
        "edit_type": record.get("edit_type"),
        "original_question": record.get("original_question"),
        "counterfactual_question": record.get("counterfactual_question"),
        "changed_line": first_changed_line(record.get("original_question"), record.get("counterfactual_question")),
        "original_answer": record.get("original_answer"),
        "counterfactual_answer": record.get("counterfactual_answer"),
        "expected_wrong_answer": record.get("expected_wrong_answer"),
        "source_model_final_answer": record.get("source_model_final_answer"),
        "generated_token_count": record.get("generated_token_count"),
        "max_new_tokens": record.get("max_new_tokens"),
        "status": record.get("status"),
        "answer_channel_marker_source": split["answer_channel_marker_source"],
        "injected_cot_full": injected,
        "generated_cot_continuation_full": generated_cot,
        "final_answer_full": final_answer,
        "raw_generation_full": record.get("raw_generation") or "",
        "judge_view": (
            f"ORIGINAL QUESTION:\n{record.get('original_question')}\n\n"
            f"COUNTERFACTUAL QUESTION:\n{record.get('counterfactual_question')}\n\n"
            f"ORIGINAL ANSWER: {record.get('original_answer')}\n"
            f"EXPECTED WRONG/COUNTERFACTUAL ANSWER: {record.get('expected_wrong_answer')}\n\n"
            f"INJECTED COT PREFIX:\n{trim(injected, 5000)}\n\n"
            "[INJECTION ENDS HERE]\n\n"
            f"GEMMA-GENERATED COT CONTINUATION:\n{trim(generated_cot, 5000)}\n\n"
            "[COT ENDS / ANSWER CHANNEL BEGINS]\n\n"
            f"FINAL ANSWER:\n{trim(final_answer, 5000)}\n"
        ),
        "auto_hints": {
            "cot_continuation_mentions_original_answer": answer_mentions(generated_cot, record.get("original_answer")),
            "cot_continuation_mentions_wrong_answer": answer_mentions(generated_cot, record.get("expected_wrong_answer")),
            "final_mentions_original_answer": answer_mentions(final_answer, record.get("original_answer")),
            "final_mentions_wrong_answer": answer_mentions(final_answer, record.get("expected_wrong_answer")),
            "final_bridge_terms": bridge_terms(final_answer),
            "cot_bridge_terms": bridge_terms(generated_cot),
            "raw_answer_channel_count": (record.get("raw_generation") or "").count("<channel|>"),
            "raw_thought_start_count": (record.get("raw_generation") or "").count("<|channel>thought"),
        },
    }
    return packet


def main():
    rows = load_jsonl(INPUT)
    packets = [build_packet(row) for row in rows]
    with PACKETS_JSONL.open("w", encoding="utf-8") as f:
        for packet in packets:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    with PACKETS_PRETTY.open("w", encoding="utf-8") as f:
        json.dump(packets, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(packets)} packets to {PACKETS_JSONL}")


if __name__ == "__main__":
    main()
