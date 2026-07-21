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
        kwargs["dtype"] = dtype
    if profile["adapter"] == "gemma4":
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "auto"
    try:
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(profile["model_id"], **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(profile["model_id"], **kwargs)
        try:
            model.config._attn_implementation = "sdpa"
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


def source_spans(row: dict[str, Any], adapter: Any, boundary_ids: list[int]) -> dict[str, Any]:
    question_start, question_end = find_question_span(row, adapter)
    trace_start = len(row["prompt_token_ids"])
    trace_end = max(trace_start, len(boundary_ids) - len(adapter.markers.answer_boundary))
    structural_positions = []
    structural_ids = set(
        adapter.markers.reasoning_open
        + adapter.markers.answer_boundary
        + adapter.markers.terminal_ids
    )
    for index, token_id in enumerate(boundary_ids):
        if int(token_id) in structural_ids:
            structural_positions.append(index)
    return {
        "question": [question_start, question_end],
        "trace": [trace_start, trace_end],
        "protected_structural_positions": structural_positions,
        "protected_structural_token_ids": sorted(int(value) for value in structural_ids),
    }


def make_source_mask(
    length: int,
    row: dict[str, Any],
    adapter: Any,
    mask_condition: str,
    boundary_ids: list[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mask_condition not in {"none", "question", "trace", "joint"}:
        raise ValueError(f"Unknown source-mask condition: {mask_condition}")
    if length < len(boundary_ids):
        raise ValueError("Attention-mask length cannot be shorter than the frozen boundary")
    spans = source_spans(row, adapter, boundary_ids)
    protected = set(spans["protected_structural_positions"])
    mask = torch.ones(length, dtype=torch.long)
    selected_spans: list[list[int]] = []
    if mask_condition in {"question", "joint"}:
        selected_spans.append(spans["question"])
    if mask_condition in {"trace", "joint"}:
        selected_spans.append(spans["trace"])
    masked_positions: list[int] = []
    for start, end in selected_spans:
        for index in range(max(0, int(start)), min(len(boundary_ids), int(end))):
            if index not in protected:
                mask[index] = 0
                masked_positions.append(index)
    provenance = {
        "mask_condition": mask_condition,
        "question_span": spans["question"],
        "trace_span": spans["trace"],
        "protected_structural_positions": spans["protected_structural_positions"],
        "protected_structural_token_ids": spans["protected_structural_token_ids"],
        "masked_position_count": len(masked_positions),
        "masked_positions": masked_positions,
        "initial_prefix_mask_applied": True,
        "prefix_mask_sha256": sha256_ints(mask[: len(boundary_ids)].tolist()),
    }
    return mask, provenance


def parse_continuation(
    adapter: Any,
    row: dict[str, Any],
    generated_ids: list[int],
    max_new_tokens: int,
    first_endpoint_tokens: int = 192,
    backend_finish_reason: str | None = None,
    backend_stop_reason: Any = None,
) -> dict[str, Any]:
    answer_ids: list[int] = []
    terminal_ids = set(adapter.markers.terminal_ids)
    registered_terminal = None
    for value in generated_ids:
        if int(value) in terminal_ids:
            registered_terminal = int(value)
            break
        answer_ids.append(int(value))
    if registered_terminal is not None:
        stop_reason = "registered_terminal"
    elif len(generated_ids) >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "backend_stop_without_registered_terminal"
    text = decode(adapter, answer_ids)
    endpoint_ids = answer_ids[: int(first_endpoint_tokens)]
    return {
        "generated_token_ids": [int(value) for value in generated_ids],
        "generated_token_count": len(generated_ids),
        "final_answer_token_ids": answer_ids,
        "final_answer": text.strip(),
        "first_endpoint_token_limit": int(first_endpoint_tokens),
        "first_endpoint_token_ids": endpoint_ids,
        "first_endpoint_text": decode(adapter, endpoint_ids).strip(),
        "registered_terminal_token_id": registered_terminal,
        "truncated": stop_reason == "max_new_tokens",
        "stop_reason": stop_reason,
        "backend_finish_reason": backend_finish_reason,
        "backend_stop_reason": backend_stop_reason,
    }


def generate_source_mask_variants(
    model: Any,
    adapter: Any,
    row: dict[str, Any],
    boundary_ids: list[int],
    mask_conditions: list[str],
    max_new_tokens: int,
    first_endpoint_tokens: int,
) -> dict[str, dict[str, Any]]:





    device = device_of(model)
    batch_size = len(mask_conditions)
    prefix = torch.tensor([boundary_ids] * batch_size, dtype=torch.long, device=device)
    initial_masks = []
    provenances = []
    for condition in mask_conditions:
        mask, provenance = make_source_mask(
            len(boundary_ids), row, adapter, condition, boundary_ids
        )
        initial_masks.append(mask)
        provenances.append(provenance)
    running_mask = torch.stack(initial_masks, dim=0).to(device)
    generated: list[list[int]] = [[] for _ in mask_conditions]
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    terminal_ids = set(int(value) for value in adapter.markers.terminal_ids)
    pad_id = int(adapter.markers.pad_token_id)

    try:
        with torch.inference_mode():
            output = model(
                input_ids=prefix,
                attention_mask=running_mask,
                use_cache=True,
            )
            past = output.past_key_values
            next_ids = output.logits[:, -1, :].argmax(dim=-1)
        del output, prefix

        for step in range(max_new_tokens):
            active_before = active.clone()
            for index in range(batch_size):
                if bool(active_before[index]):
                    token_id = int(next_ids[index].item())
                    generated[index].append(token_id)
                    if token_id in terminal_ids:
                        active[index] = False
            if step + 1 >= max_new_tokens or not bool(active.any()):
                break

            visible_column = active_before.long().unsqueeze(1)
            running_mask = torch.cat([running_mask, visible_column], dim=1)
            model_tokens = torch.where(
                active_before,
                next_ids,
                torch.full_like(next_ids, pad_id),
            ).unsqueeze(1)
            with torch.inference_mode():
                output = model(
                    input_ids=model_tokens,
                    attention_mask=running_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                next_ids = output.logits[:, -1, :].argmax(dim=-1)
            del output, model_tokens
    except Exception as exc:
        raise RuntimeError(
            f"source masking failed for {row['diagnostic_parent_id']}: {exc}"
        ) from exc

    results: dict[str, dict[str, Any]] = {}
    for condition, token_ids, provenance in zip(
        mask_conditions, generated, provenances
    ):
        parsed = parse_continuation(
            adapter,
            row,
            token_ids,
            max_new_tokens,
            first_endpoint_tokens,
            backend_finish_reason="greedy_transformers",
            backend_stop_reason=None,
        )
        parsed["mask_provenance"] = provenance
        results[condition] = parsed
    return results
