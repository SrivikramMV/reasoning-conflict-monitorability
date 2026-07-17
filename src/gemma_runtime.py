

from __future__ import annotations

import gc
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from io_utils import sha256_ints


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0")


def load_gemma(config: dict[str, Any], token: str | None = None) -> tuple[Any, Any]:
    model_config = config["model"]
    dtype_name = model_config["dtype"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the final Gemma run.")
    torch.manual_seed(int(config["reproducibility"]["seed"]))
    torch.cuda.manual_seed_all(int(config["reproducibility"]["seed"]))
    torch.set_float32_matmul_precision("high")
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        token=token,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def runtime_metadata(model: Any, processor: Any) -> dict[str, Any]:
    import accelerate
    import safetensors
    import transformers

    text_config = getattr(model.config, "text_config", model.config)
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": accelerate.__version__,
        "safetensors_version": safetensors.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": processor.tokenizer.__class__.__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "num_hidden_layers": int(text_config.num_hidden_layers),
        "hidden_size": int(text_config.hidden_size),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    }


def build_prompt_ids(
    processor: Any,
    question: str,
    system_prompt: str,
    enable_thinking: bool = True,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    ids = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return [int(value) for value in ids[0].tolist()]


def response_tokens(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    header_ids = tokenizer.encode("<|channel>thought\n", add_special_tokens=False)
    values = {
        "thought_header_ids": [int(value) for value in header_ids],
        "channel_start_id": int(tokenizer.convert_tokens_to_ids("<|channel>")),
        "channel_end_id": int(tokenizer.convert_tokens_to_ids("<channel|>")),
        "turn_end_id": int(tokenizer.convert_tokens_to_ids("<turn|>")),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    if any(value < 0 for key, value in values.items() if key.endswith("_id")):
        raise ValueError(f"Missing Gemma response token: {values}")
    return values


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("Cannot locate an empty subsequence")
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    raise ValueError("Token subsequence was not found")


def first_index(values: list[int], candidates: set[int], start: int = 0) -> int | None:
    for index in range(start, len(values)):
        if values[index] in candidates:
            return index
    return None


def parse_response_ids(
    processor: Any,
    response_ids: list[int],
    generated_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    tokens = response_tokens(processor)
    header = tokens["thought_header_ids"]
    try:
        header_start, content_start = find_subsequence(response_ids, header)
    except ValueError:
        header_start = first_index(response_ids, {tokens["channel_start_id"]})
        content_start = None if header_start is None else header_start + 1

    channel_end = None
    if content_start is not None:
        channel_end = first_index(response_ids, {tokens["channel_end_id"]}, content_start)
    stop_index = first_index(
        response_ids,
        {tokens["turn_end_id"], tokens["eos_token_id"]},
        0 if channel_end is None else channel_end + 1,
    )
    response_stop = len(response_ids) if stop_index is None else stop_index

    if content_start is None:
        thought_ids: list[int] = []
        answer_ids = response_ids[:response_stop]
        parse_status = "missing_thought_header"
    elif channel_end is None:
        thought_ids = response_ids[content_start:response_stop]
        answer_ids = []
        parse_status = "missing_answer_boundary"
    else:
        thought_ids = response_ids[content_start:channel_end]
        answer_ids = response_ids[channel_end + 1 : response_stop]
        parse_status = "ok"

    stop_reason = "max_new_tokens"
    if stop_index is not None:
        stop_reason = "turn_end" if response_ids[stop_index] == tokens["turn_end_id"] else "eos"
    elif len(generated_ids) < max_new_tokens:
        stop_reason = "generation_stopped_without_registered_token"

    tokenizer = processor.tokenizer
    return {
        "parse_status": parse_status,
        "thought_header_start": header_start,
        "answer_boundary_response_index": channel_end,
        "response_stop_index": response_stop,
        "thought_token_ids": thought_ids,
        "final_answer_token_ids": answer_ids,
        "thought": tokenizer.decode(
            thought_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).strip(),
        "final_answer": tokenizer.decode(
            answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).strip(),
        "raw_response": tokenizer.decode(
            response_ids[:response_stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "generated_continuation": tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_new_tokens",
    }


def generation_stop_ids(processor: Any) -> list[int]:
    tokens = response_tokens(processor)
    return [tokens["eos_token_id"], tokens["turn_end_id"]]


def generate_from_ids(
    model: Any,
    processor: Any,
    input_ids: list[int],
    response_prefix_ids: list[int],
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    device = model_device(model)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    kwargs: dict[str, Any] = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "eos_token_id": generation_stop_ids(processor),
        "pad_token_id": response_tokens(processor)["pad_token_id"],
        "use_cache": True,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "generator": generator,
            }
        )
    started = time.time()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    elapsed = time.time() - started
    generated = [int(value) for value in output[0, len(input_ids) :].tolist()]
    response_ids = list(response_prefix_ids) + generated
    parsed = parse_response_ids(processor, response_ids, generated, max_new_tokens)
    parsed.update(
        {
            "generated_token_ids": generated,
            "generated_token_count": len(generated),
            "input_token_count": len(input_ids),
            "input_token_sha256": sha256_ints(input_ids),
            "elapsed_seconds": elapsed,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
    )
    del output, input_tensor, attention_mask
    return parsed


def token_safe_cut(
    tokenizer: Any,
    thought_ids: list[int],
    ratio: float,
) -> dict[str, Any]:
    if not 0 < ratio < 1:
        raise ValueError("Partial injection ratios must lie strictly between zero and one")
    count = len(thought_ids)
    if count < 12:
        raise ValueError("Source thought is too short for a partial intervention")
    target = max(4, min(count - 2, round(count * ratio)))
    radius = max(8, round(count * 0.12))
    lower = max(4, target - radius)
    upper = min(count - 1, target + radius)
    candidates: list[tuple[int, int, int, str]] = []
    for cut in range(lower, upper + 1):
        decoded = tokenizer.decode(
            thought_ids[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        stripped = decoded.rstrip()
        if re.search(r"(?:[.!?](?:[\"')\]]*)|\n)\s*$", decoded):
            quality = 0
        elif decoded.endswith((" ", "\n", "\t")):
            quality = 1
        elif stripped.endswith((",", ";", ":")):
            quality = 2
        else:
            quality = 3
        candidates.append((quality, abs(cut - target), cut, decoded))
    quality, distance, cut, decoded = min(candidates, key=lambda value: (value[0], value[1]))
    labels = {0: "sentence_or_line_boundary", 1: "whitespace_boundary", 2: "clause_boundary", 3: "token_boundary"}
    return {
        "cut_token_count": cut,
        "source_thought_token_count": count,
        "target_ratio": ratio,
        "actual_ratio": cut / count,
        "target_token_count": target,
        "distance_from_target_tokens": distance,
        "boundary_quality": labels[quality],
        "decoded_prefix": decoded,
    }


def build_injected_prefix(
    processor: Any,
    source_thought_ids: list[int],
    ratio: float,
    close_thought_channel: bool,
) -> tuple[list[int], dict[str, Any]]:
    tokens = response_tokens(processor)
    header = tokens["thought_header_ids"]
    if ratio >= 1:
        selected = list(source_thought_ids)
        cut = {
            "cut_token_count": len(selected),
            "source_thought_token_count": len(selected),
            "target_ratio": 1.0,
            "actual_ratio": 1.0,
            "target_token_count": len(selected),
            "distance_from_target_tokens": 0,
            "boundary_quality": "complete_trace",
            "decoded_prefix": processor.tokenizer.decode(
                selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        }
    else:
        cut = token_safe_cut(processor.tokenizer, source_thought_ids, ratio)
        selected = source_thought_ids[: cut["cut_token_count"]]
    prefix = list(header) + selected
    if close_thought_channel:
        prefix.append(tokens["channel_end_id"])
    return prefix, cut


def continuation_logprob_ids(
    model: Any,
    prefix_ids: list[int],
    continuation_ids: list[int],
) -> dict[str, Any]:
    if not continuation_ids:
        raise ValueError("Continuation produced no tokens")
    device = model_device(model)
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


def completed_boundary_ids(
    processor: Any,
    prompt_ids: list[int],
    injected_ids: list[int],
    generated_ids: list[int],
) -> list[int]:
    tokens = response_tokens(processor)
    response = list(injected_ids) + list(generated_ids)
    boundary_index = first_index(response, {tokens["channel_end_id"]})
    if boundary_index is None:
        raise ValueError("No final-answer boundary in the intervention response")
    return list(prompt_ids) + response[: boundary_index + 1]


def sampled_generate_batch(
    model: Any,
    processor: Any,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    count: int,
    seed: int,
) -> list[list[int]]:
    device = model_device(model)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(
            input_ids=tensor,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=count,
            eos_token_id=generation_stop_ids(processor),
            pad_token_id=response_tokens(processor)["pad_token_id"],
            use_cache=True,
        )
    rows = [[int(value) for value in sequence[len(input_ids) :].tolist()] for sequence in output]
    del output, tensor, mask
    return rows


def text_layers(model: Any) -> Any:
    root = model.model
    if hasattr(root, "language_model"):
        root = root.language_model
    if not hasattr(root, "layers"):
        raise AttributeError("Could not locate Gemma text layers")
    return root.layers


def text_norm(model: Any) -> Any:
    root = model.model
    if hasattr(root, "language_model"):
        root = root.language_model
    return root.norm


def capture_hidden_positions(
    model: Any,
    token_ids: list[int],
    positions: list[int],
) -> np.ndarray:
    layers = text_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):
        def hook(module: Any, args: tuple[Any, ...], layer_index: int = index) -> None:
            captured[layer_index] = args[0][0, positions, :].detach().to(torch.float16).cpu()

        handles.append(layer.register_forward_pre_hook(hook))
    final: dict[str, torch.Tensor] = {}

    def norm_hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        final["state"] = output[0, positions, :].detach().to(torch.float16).cpu()

    handles.append(text_norm(model).register_forward_hook(norm_hook))
    device = model_device(model)
    tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    try:
        with torch.inference_mode():
            output = model(input_ids=tensor, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(layers) or "state" not in final:
        raise RuntimeError("Hidden-state hooks did not capture every requested depth")
    ordered = [captured[index].numpy() for index in range(len(layers))]
    ordered.append(final["state"].numpy())
    result = np.stack(ordered, axis=0)
    del output, tensor, mask
    return result


def question_token_span(processor: Any, prompt_ids: list[int], question: str) -> tuple[int, int]:
    question_ids = processor.tokenizer.encode(question, add_special_tokens=False)
    return find_subsequence(prompt_ids, [int(value) for value in question_ids])


def make_source_attention_mask(
    length: int,
    question_span: tuple[int, int],
    cot_span: tuple[int, int],
    condition: str,
    prefix_ids: list[int],
    special_ids: set[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((1, length), dtype=torch.long, device=device)
    if condition in {"question", "question_and_cot"}:
        mask[:, question_span[0] : question_span[1]] = 0
    if condition in {"cot", "question_and_cot"}:
        for position in range(cot_span[0], min(cot_span[1], len(prefix_ids))):
            if prefix_ids[position] not in special_ids:
                mask[:, position] = 0
    return mask


def masked_answer_generate(
    model: Any,
    processor: Any,
    boundary_ids: list[int],
    question_span: tuple[int, int],
    cot_span: tuple[int, int],
    condition: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = model_device(model)
    prefix = torch.tensor([boundary_ids], dtype=torch.long, device=device)
    prefix_length = prefix.shape[-1]
    with torch.inference_mode():
        prefill = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
    cache = prefill.past_key_values
    stop_ids = set(generation_stop_ids(processor))
    first = int(torch.argmax(prefill.logits[0, -1]).item())
    generated = [] if first in stop_ids else [first]
    current = torch.tensor([[first]], dtype=torch.long, device=device)
    special_ids = set(int(value) for value in processor.tokenizer.all_special_ids)
    del prefill

    for _ in range(max(0, max_new_tokens - len(generated))):
        if not generated:
            break
        total_length = prefix_length + len(generated)
        attention = make_source_attention_mask(
            total_length,
            question_span,
            cot_span,
            condition,
            boundary_ids,
            special_ids,
            device,
        )
        cache_position = torch.tensor([total_length - 1], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(
                input_ids=current,
                attention_mask=attention,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
        cache = output.past_key_values
        next_token = int(torch.argmax(output.logits[0, -1]).item())
        del output, attention, cache_position
        if next_token in stop_ids:
            break
        generated.append(next_token)
        current = torch.tensor([[next_token]], dtype=torch.long, device=device)

    text = processor.tokenizer.decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    del cache, prefix, current
    return {"generated_token_ids": generated, "generated_token_count": len(generated), "text": text}


def set_attention_implementation(model: Any, implementation: str) -> str | None:
    previous = getattr(model.config, "_attn_implementation", None)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(implementation)
    else:
        model.config._attn_implementation = implementation
    return previous


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
