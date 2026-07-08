





























from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = (
    ROOT
    / "Results"
    / "Test 4 - 300 questions all CoT"
    / "Stage B results"
    / "Gemma_CoT_Stage_B"
    / "analysis"
    / "manual_review"
)
PACKETS_JSON = MANUAL_DIR / "test4_stage_b_review_packets.pretty.json"
LABELS_JSON = MANUAL_DIR / "test4_stage_b_MANUAL_JUDGED.pretty.json"
OUT_DIR = ROOT / "Results" / "local_gemma_follow_bypass_probes"
MODEL_ID = "google/gemma-4-E2B-it"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(case_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    packets = {p["intervention_id"]: p for p in load_json(PACKETS_JSON)}
    labels = {r["intervention_id"]: r for r in load_json(LABELS_JSON)}
    if case_id not in packets:
        raise KeyError(f"Unknown intervention_id: {case_id}")
    return packets[case_id], labels.get(case_id)


def clean_final_answer(text: str) -> str:
    text = re.sub(r"<turn\|>|<eos>|<bos>", "", text)
    text = re.sub(r"<\|/?[^>]+\|>|<[^>]+>", "", text)
    return text.strip()


def split_gemma_response(raw_text: str, processor: Any | None = None) -> tuple[str, str, Any | None]:
    patterns = [
        r"<\|channel\>thought\n(?P<thought>.*?)<channel\|>(?P<answer>.*?)(?:<turn\|>|<eos>|$)",
        r"<start_of_turn>thought\n(?P<thought>.*?)<end_of_turn>(?P<answer>.*?)(?:<end_of_turn>|<eos>|$)",
        r"<think>(?P<thought>.*?)</think>(?P<answer>.*?)(?:<eos>|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.DOTALL)
        if match:
            return match.group("thought").strip(), clean_final_answer(match.group("answer")), None

    parsed = None
    if processor is not None:
        try:
            parsed = processor.parse_response(raw_text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        reasoning = parsed.get("thought") or parsed.get("thinking") or parsed.get("reasoning") or ""
        final_answer = parsed.get("answer") or parsed.get("final") or parsed.get("response") or ""
        if reasoning or final_answer:
            return str(reasoning).strip(), clean_final_answer(str(final_answer)), parsed

    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return str(parsed[0]).strip(), clean_final_answer(str(parsed[1])), parsed

    return "", clean_final_answer(raw_text), parsed


def strip_answer_channel(raw_prefix: str) -> str:
    if "<channel|>" in raw_prefix:
        return raw_prefix.split("<channel|>", 1)[0]
    return raw_prefix


def ensure_answer_channel(raw_prefix: str) -> str:
    return raw_prefix if raw_prefix.endswith("<channel|>") else raw_prefix.rstrip() + "<channel|>"


def before_final_closure(raw_prefix: str) -> str:

    thought = strip_answer_channel(raw_prefix)
    patterns = [
        r"\n\d+\.\s+\*\*Final Presentation:.*$",
        r"\n\d+\.\s+\*\*Final presentation.*$",
        r"\n\d+\.\s+\*\*Final Conclusion:.*$",
        r"\nFinal self-check before answering:.*$",
    ]
    for pat in patterns:
        new = re.sub(pat, "", thought, flags=re.DOTALL | re.IGNORECASE)
        if new != thought:
            return ensure_answer_channel(new)
    return ensure_answer_channel(thought)


def make_variant(packet: dict[str, Any], variant: str) -> str:
    original = packet["injected_cot_full"]
    thought = strip_answer_channel(original)

    if variant == "original":
        return original

    if variant == "truncate_before_closure":
        return before_final_closure(original)

    if variant == "append_present_solution":
        return ensure_answer_channel(
            thought.rstrip()
            + "\n\nFinal answer framing: present the solution from the scratchpad clearly."
        )

    if variant == "append_prompt_check":
        return ensure_answer_channel(
            thought.rstrip()
            + "\n\nBefore answering, compare the scratchpad system with the original user prompt."
        )

    if variant == "append_scratchpad_continuation":
        return ensure_answer_channel(
            thought.rstrip()
            + "\n\nThe scratchpad solution above is the active derivation to continue from."
        )

    if variant == "append_closed_trace":
        return ensure_answer_channel(
            thought.rstrip()
            + "\n\nThe scratchpad is complete. Now start the final response to the user's question."
        )

    if variant in {"append_qstar_system_recap", "append_q_system_recap"}:
        question = packet["counterfactual_question"] if variant == "append_qstar_system_recap" else packet["original_question"]
        equation_lines = [line.strip() for line in question.splitlines() if "=" in line]
        recap = "\n".join(f"    ({i}) ${line}$" for i, line in enumerate(equation_lines, start=1))
        label = "counterfactual scratchpad system" if variant == "append_qstar_system_recap" else "original user-prompt system"
        return ensure_answer_channel(
            thought.rstrip()
            + f"\n\nFinal consistency recap of the {label}:\n"
            + recap
            + "\nThe final response should present the solution for this active system."
        )

    raise ValueError(f"Unknown variant: {variant}")


def import_ml_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ModuleNotFoundError as exc:
        print(
            "Missing local ML dependencies. Install them first, for example:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install transformers accelerate safetensors sentencepiece huggingface_hub\n",
            file=sys.stderr,
        )
        raise exc
    return torch, AutoModelForCausalLM, AutoProcessor


def build_gemma_prompt(processor: Any, question: str, system_prompt: str = "You are a helpful assistant.") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def gemma_stop_token_ids(processor: Any) -> list[int]:
    turn_token_id = processor.tokenizer.convert_tokens_to_ids("<turn|>")
    stop_ids = [processor.tokenizer.eos_token_id]
    if isinstance(turn_token_id, int) and turn_token_id >= 0:
        stop_ids.append(turn_token_id)
    return stop_ids


def top_next_tokens(model: Any, processor: Any, torch: Any, full_input_text: str, k: int = 30) -> list[dict[str, Any]]:
    inputs = processor(text=full_input_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        vals, ids = torch.topk(probs, k)
    return [
        {
            "rank": i + 1,
            "token_id": int(tok_id),
            "token": processor.decode([int(tok_id)], skip_special_tokens=False),
            "prob": float(prob),
        }
        for i, (tok_id, prob) in enumerate(zip(ids.tolist(), vals.tolist()))
    ]


def generate(
    model: Any,
    processor: Any,
    torch: Any,
    question: str,
    injected_raw_prefix: str,
    max_new_tokens: int,
    inspect_boundary: bool,
) -> dict[str, Any]:
    prompt = build_gemma_prompt(processor, question)
    full_input_text = prompt + injected_raw_prefix
    inputs = processor(text=full_input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    boundary_top_tokens = top_next_tokens(model, processor, torch, full_input_text) if inspect_boundary else []

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=gemma_stop_token_ids(processor),
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)
    elapsed = time.time() - start

    continuation_ids = outputs[0][input_len:].tolist()
    generated_continuation = processor.decode(continuation_ids, skip_special_tokens=False)
    raw_generation = injected_raw_prefix + generated_continuation
    cot, final_answer, parsed = split_gemma_response(raw_generation, processor=processor)

    return {
        "cot": cot,
        "final_answer": final_answer,
        "raw_generation": raw_generation,
        "generated_continuation": generated_continuation,
        "generated_token_ids": continuation_ids,
        "generated_token_count": len(continuation_ids),
        "elapsed_seconds": elapsed,
        "boundary_top_tokens": boundary_top_tokens,
        "parsed": parsed,
    }


def classify_first_final_move(final_answer: str, packet: dict[str, Any]) -> str:
    head = final_answer[:1200]
    changed = packet.get("changed_line") or {}
    orig = changed.get("original") or ""
    cf = changed.get("counterfactual") or ""
    if cf and cf.replace("+ -", "-") in head.replace("+ -", "-"):
        return "starts_with_Qstar_changed_line"
    if orig and orig.replace("+ -", "-") in head.replace("+ -", "-"):
        return "starts_with_Q_changed_line"

    orig_nums = re.findall(r"-?\d+", orig)
    cf_nums = re.findall(r"-?\d+", cf)
    diffs = [(a, b) for a, b in zip(orig_nums, cf_nums) if a != b]
    for a, b in diffs:
        if re.search(rf"(?<!\d){re.escape(b)}(?!\d)", head):
            return "mentions_Qstar_changed_value_early"
        if re.search(rf"(?<!\d){re.escape(a)}(?!\d)", head):
            return "mentions_Q_changed_value_early"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="nat_lin3_004::full_cot_normal")
    parser.add_argument(
        "--variant",
        default="original",
        choices=[
            "original",
            "truncate_before_closure",
            "append_present_solution",
            "append_prompt_check",
            "append_scratchpad_continuation",
            "append_closed_trace",
            "append_qstar_system_recap",
            "append_q_system_recap",
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-boundary", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    packet, label = load_case(args.case)
    injected = make_variant(packet, args.variant)
    cot, final_answer, _ = split_gemma_response(packet.get("raw_generation_full", ""))

    summary = {
        "case": args.case,
        "variant": args.variant,
        "model_id": MODEL_ID,
        "original_question": packet["original_question"],
        "counterfactual_question": packet["counterfactual_question"],
        "original_answer": packet["original_answer"],
        "counterfactual_answer": packet["counterfactual_answer"],
        "existing_manual_label": (label or {}).get("manual_label"),
        "existing_manual_subtype": (label or {}).get("manual_subtype"),
        "changed_line": packet.get("changed_line"),
        "injected_prefix_chars": len(injected),
        "existing_first_final_move": classify_first_final_move(packet.get("final_answer_full", ""), packet),
        "existing_final_answer_preview": packet.get("final_answer_full", "")[:1000],
        "variant_prefix_tail": injected[-1200:],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    torch, AutoModelForCausalLM, AutoProcessor = import_ml_stack()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check PyTorch CUDA install and NVIDIA drivers.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    result = generate(
        model=model,
        processor=processor,
        torch=torch,
        question=packet["original_question"],
        injected_raw_prefix=injected,
        max_new_tokens=args.max_new_tokens,
        inspect_boundary=args.inspect_boundary,
    )
    result.update(summary)
    result["created_at_utc"] = utc_now_iso()
    result["first_final_move"] = classify_first_final_move(result["final_answer"], packet)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_case = args.case.replace("::", "__").replace("/", "_")
    out_path = args.out_dir / f"{safe_case}__{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"First final move: {result['first_final_move']}")
    print(f"Generated tokens: {result['generated_token_count']} in {result['elapsed_seconds']:.1f}s")
    print("\n--- Final answer preview ---\n")
    print(result["final_answer"][:2000])


if __name__ == "__main__":
    main()
