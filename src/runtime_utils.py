from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from adapters import make_adapter
from classification import classify_visible_branch, equation_lines
from common import JsonlCheckpoint, atomic_write_json, find_subsequence, read_json, read_jsonl, sha256_file, sha256_ints, stable_seed, utc_now


CANDIDATE_TEMPLATES = {
    "standard_system": {
        2: "This is a system of two linear equations. We can solve it using elimination.\n\nThe system is:\n",
        3: "This is a system of three linear equations. We can solve it using elimination.\n\nThe system is:\n",
    },
    "plain_restatement": {2: "The equations are:\n", 3: "The equations are:\n"},
    "active_system": {
        2: "I will solve the following active system:\n",
        3: "I will solve the following active system:\n",
    },
    "verification": {
        2: "Checking the problem statement, the equations are:\n",
        3: "Checking the problem statement, the equations are:\n",
    },
    "minimal": {2: "", 3: ""},
}


def read_pairs(bundle_root: Path) -> dict[str, dict[str, Any]]:
    return {row["pair_id"]: row for row in read_jsonl(bundle_root / "data" / "frozen_common_linear_pairs_29.jsonl")}


def ensure_cuda_a100(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this package.")
    gpu_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    runtime = config["runtime"]
    if runtime["required_gpu_name_fragment"].upper() not in gpu_name.upper():
        raise RuntimeError(f"Expected an A100 runtime, found {gpu_name}")
    if memory < int(runtime["minimum_gpu_memory_bytes"]):
        raise RuntimeError(f"GPU memory is below the frozen minimum: {memory} bytes")
    return {"gpu_name": gpu_name, "gpu_total_memory_bytes": int(memory)}


def dtype_from_profile(profile: dict[str, Any]):
    if profile.get("dtype") == "bfloat16":
        return torch.bfloat16
    if profile.get("dtype") == "float16":
        return torch.float16
    return "auto"


def load_tokenizer_or_processor(profile: dict[str, Any]):
    token = os.getenv("HF_TOKEN")
    if profile["adapter"] == "gemma4":
        processor = AutoProcessor.from_pretrained(profile["model_id"], revision=profile["revision"], token=token)
        return processor
    tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], revision=profile["revision"], token=token)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_transformers_model(profile: dict[str, Any]):
    token = os.getenv("HF_TOKEN")
    processor_or_tokenizer = load_tokenizer_or_processor(profile)
    kwargs = {
        "revision": profile["revision"],
        "token": token,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    dtype = dtype_from_profile(profile)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if profile["adapter"] == "gemma4":
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "auto"
    try:
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(profile["model_id"], **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(profile["model_id"], **kwargs)
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass
    model.eval()
    return processor_or_tokenizer, model, make_adapter(profile["adapter"], processor_or_tokenizer, profile)


def cleanup_model(model: Any | None = None) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def device_of(model: Any) -> torch.device:
    return next(model.parameters()).device


def decode(adapter: Any, ids: list[int]) -> str:
    return adapter.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def completed_boundary_ids(row: dict[str, Any], adapter: Any) -> list[int]:
    response_prefix = [int(v) for v in row["response_prefix_token_ids"]]
    generated = [int(v) for v in row["generated_token_ids"]]
    response = response_prefix + generated
    marker = adapter.markers.answer_boundary
    start = row.get("answer_boundary_response_index")
    if start is None:
        start, end = find_subsequence(response, marker)
    else:
        end = int(start) + len(marker)
        if response[int(start):end] != marker:
            start, end = find_subsequence(response, marker)
    generated_take = max(0, end - len(response_prefix))
    return [int(v) for v in row["input_token_ids"]] + generated[:generated_take]


def build_candidate_texts(pair: dict[str, Any], template: str) -> tuple[str, str, str]:
    original = equation_lines(pair["original_question"])
    counterfactual = equation_lines(pair["counterfactual_question"])
    changed = [index for index, values in enumerate(zip(original, counterfactual)) if values[0] != values[1]]
    if len(changed) != 1:
        raise ValueError(f"Expected one changed equation in {pair['pair_id']}")
    changed_index = changed[0]
    variable_count = len(pair["structured_problem"]["variables"])
    preamble = CANDIDATE_TEMPLATES[template][variable_count]
    for index, equation in enumerate(original[:changed_index], start=1):
        preamble += f"{index}) {equation}\n"
    number = changed_index + 1
    return preamble, f"{number}) {original[changed_index]}\n", f"{number}) {counterfactual[changed_index]}\n"


def continuation_logprob_ids(model: Any, prefix_ids: list[int], continuation_ids: list[int]) -> dict[str, Any]:
    if not continuation_ids:
        raise ValueError("Continuation produced no tokens")
    device = device_of(model)
    full = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(full)
    prefix_length = len(prefix_ids)
    with torch.inference_mode():
        output = model(input_ids=full, attention_mask=mask, use_cache=False)
        logits = output.logits[:, prefix_length - 1 : -1, :].float()
        targets = full[:, prefix_length:]
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    values = selected[0].detach().cpu().tolist()
    total = float(sum(values))
    count = len(values)
    del output, full, mask, logits, targets, log_probs, selected
    return {
        "token_count": count,
        "total_logprob": total,
        "average_logprob": total / count,
        "perplexity": math.exp(-total / count),
        "token_logprobs": values,
    }


def locate_layers_and_norm(model: Any) -> tuple[list[Any], Any | None]:
    roots = [getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None), getattr(model, "language_model", None)]
    for root in roots:
        if root is not None and hasattr(root, "layers"):
            return list(root.layers), getattr(root, "norm", None)
    raise RuntimeError(f"Could not locate transformer layers on {model.__class__.__name__}")


def capture_hidden_positions(model: Any, ids: list[int], positions: list[int]) -> tuple[np.ndarray, list[int]]:
    layers, norm = locate_layers_and_norm(model)
    device = device_of(model)
    tensors: list[torch.Tensor | None] = [None] * (len(layers) + (1 if norm is not None else 0))
    hooks = []

    def layer_hook(index: int):
        def hook(_module, inputs):
            tensors[index] = inputs[0][0, positions, :].detach().float().cpu()
        return hook

    for index, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(layer_hook(index)))
    if norm is not None:
        def norm_hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            tensors[len(layers)] = value[0, positions, :].detach().float().cpu()
        hooks.append(norm.register_forward_hook(norm_hook))
    try:
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_tensor)
        with torch.inference_mode():
            model(input_ids=input_tensor, attention_mask=mask, use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
    if any(value is None for value in tensors):
        raise RuntimeError("At least one hidden-state hook did not fire")
    stacked = torch.stack([value for value in tensors if value is not None], dim=0).numpy()
    return stacked, positions


def find_question_span(row: dict[str, Any], adapter: Any) -> tuple[int, int]:
    prompt_ids = [int(v) for v in row["prompt_token_ids"]]
    question_ids = adapter._encode(row["original_question"])
    try:
        start, end = find_subsequence(prompt_ids, question_ids)
        return start, end
    except ValueError:
        compact = adapter._encode(row["original_question"].strip())
        start, end = find_subsequence(prompt_ids, compact)
        return start, end


def make_source_mask(length: int, row: dict[str, Any], adapter: Any, mask_condition: str, boundary_ids: list[int]) -> torch.Tensor:
    mask = torch.ones(length, dtype=torch.long)
    protected = set(adapter.markers.reasoning_open + adapter.markers.answer_boundary + adapter.markers.terminal_ids)
    spans: list[tuple[int, int]] = []
    q_span = find_question_span(row, adapter)
    trace_span = (len(row["prompt_token_ids"]), max(len(row["prompt_token_ids"]), len(boundary_ids) - len(adapter.markers.answer_boundary)))
    if mask_condition in {"question", "joint"}:
        spans.append(q_span)
    if mask_condition in {"trace", "joint"}:
        spans.append(trace_span)
    for start, end in spans:
        for index in range(max(0, start), min(length, end)):
            if int(boundary_ids[index]) not in protected:
                mask[index] = 0
    return mask


def generate_with_source_mask(model: Any, adapter: Any, row: dict[str, Any], boundary_ids: list[int], mask_condition: str, max_new_tokens: int) -> dict[str, Any]:
    device = device_of(model)
    generated: list[int] = []
    stop_ids = set(adapter.markers.terminal_ids)
    prefix_ids = list(boundary_ids)
    first_mask = torch.ones(len(prefix_ids), dtype=torch.long, device=device)
    try:
        with torch.inference_mode():
            input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            output = model(input_ids=input_tensor, attention_mask=first_mask.unsqueeze(0), use_cache=True)
            past = output.past_key_values
            next_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        del input_tensor, output
        if next_id in stop_ids:
            return parse_continuation(adapter, row, generated, max_new_tokens)
        for _ in range(max_new_tokens - 1):
            full_len = len(prefix_ids) + len(generated)
            mask = make_source_mask(full_len, row, adapter, mask_condition, boundary_ids).to(device)
            new_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(input_ids=new_token, attention_mask=mask.unsqueeze(0), past_key_values=past, use_cache=True)
                past = output.past_key_values
                next_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            del output, new_token
            if next_id in stop_ids:
                break
    except Exception as exc:
        raise RuntimeError(f"source masking failed for {row['diagnostic_parent_id']} {mask_condition}: {exc}") from exc
    return parse_continuation(adapter, row, generated, max_new_tokens)


def parse_continuation(adapter: Any, row: dict[str, Any], generated_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    answer_ids = []
    stop_reason = "max_new_tokens" if len(generated_ids) >= max_new_tokens else "generation_stopped_without_registered_token"
    for value in generated_ids:
        if value in set(adapter.markers.terminal_ids):
            stop_reason = "terminal"
            break
        answer_ids.append(int(value))
    text = decode(adapter, answer_ids)
    return {
        "generated_token_ids": [int(v) for v in generated_ids],
        "generated_token_count": len(generated_ids),
        "final_answer_token_ids": answer_ids,
        "final_answer": text.strip(),
        "truncated": stop_reason == "max_new_tokens",
        "stop_reason": stop_reason,
    }
