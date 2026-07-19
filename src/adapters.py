

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common import complete_cut, find_subsequence, sha256_ints, token_safe_cut


@dataclass(frozen=True)
class MarkerSet:
    reasoning_open: list[int]
    answer_boundary: list[int]
    terminal_ids: list[int]
    pad_token_id: int


class ReasoningAdapter:
    name = "base"
    prompt_contains_reasoning_open = False
    reasoning_open_optional = False

    def __init__(self, processor_or_tokenizer: Any, profile: dict[str, Any]) -> None:
        self.processor = processor_or_tokenizer
        self.tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        self.profile = profile
        self.markers = self._markers()

    def _encode(self, text: str) -> list[int]:
        return [int(value) for value in self.tokenizer.encode(text, add_special_tokens=False)]

    def _markers(self) -> MarkerSet:
        raise NotImplementedError

    def build_prompt_ids(self, question: str) -> list[int]:
        raise NotImplementedError

    def clean_response_prefix(self) -> list[int]:
        return list(self.markers.reasoning_open) if self.prompt_contains_reasoning_open else []

    def build_injected_input(
        self,
        prompt_ids: list[int],
        source_thought_ids: list[int],
        ratio: float,
        close_reasoning: bool,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        cut = (
            complete_cut(self.tokenizer, source_thought_ids)
            if ratio >= 1
            else token_safe_cut(self.tokenizer, source_thought_ids, ratio)
        )
        selected = source_thought_ids[: int(cut["cut_token_count"])]
        response_prefix = list(self.markers.reasoning_open) + selected
        input_suffix = selected if self.prompt_contains_reasoning_open else response_prefix
        if close_reasoning:
            response_prefix += self.markers.answer_boundary
            input_suffix += self.markers.answer_boundary
        return list(prompt_ids) + input_suffix, response_prefix, cut

    def parse_response(
        self,
        response_prefix_ids: list[int],
        generated_ids: list[int],
        max_new_tokens: int,
        finish_reason: str | None,
        stop_reason: Any,
    ) -> dict[str, Any]:
        response_ids = list(response_prefix_ids) + list(generated_ids)
        try:
            open_start, content_start = find_subsequence(response_ids, self.markers.reasoning_open)
        except ValueError:
            open_start, content_start = None, None
        boundary_start = boundary_end = None
        if content_start is not None:
            try:
                boundary_start, boundary_end = find_subsequence(
                    response_ids, self.markers.answer_boundary, start=content_start
                )
            except ValueError:
                pass
        elif self.reasoning_open_optional:
            try:
                boundary_start, boundary_end = find_subsequence(
                    response_ids, self.markers.answer_boundary, start=0
                )
                content_start = 0
            except ValueError:
                pass
        terminal_index = None
        terminal_set = set(self.markers.terminal_ids)
        for index in range(boundary_end or 0, len(response_ids)):
            if response_ids[index] in terminal_set:
                terminal_index = index
                break
        response_stop = terminal_index if terminal_index is not None else len(response_ids)
        if content_start is None:
            thought_ids: list[int] = []
            final_ids = response_ids[:response_stop]
            parse_status = "missing_reasoning_open"
        elif boundary_start is None or boundary_end is None:
            thought_ids = response_ids[content_start:response_stop]
            final_ids = []
            parse_status = "missing_answer_boundary"
        else:
            thought_ids = response_ids[content_start:boundary_start]
            final_ids = response_ids[boundary_end:response_stop]
            parse_status = "ok"
        decode = lambda ids: self.tokenizer.decode(              
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        truncated = finish_reason == "length" or (
            finish_reason is None and len(generated_ids) >= max_new_tokens
        )
        return {
            "parse_status": parse_status,
            "reasoning_open_response_index": open_start,
            "answer_boundary_response_index": boundary_start,
            "response_stop_index": response_stop,
            "thought_token_ids": thought_ids,
            "final_answer_token_ids": final_ids,
            "thought": decode(thought_ids).strip(),
            "final_answer": decode(final_ids).strip(),
            "raw_response": decode(response_ids[:response_stop]),
            "generated_continuation": decode(generated_ids),
            "generated_token_ids": [int(value) for value in generated_ids],
            "generated_token_count": len(generated_ids),
            "response_prefix_token_ids": [int(value) for value in response_prefix_ids],
            "response_prefix_sha256": sha256_ints(response_prefix_ids),
            "finish_reason": finish_reason,
            "backend_stop_reason": stop_reason,
            "truncated": truncated,
        }

    def validate_template(self, question: str = "Evaluate: 2 + 3.") -> dict[str, Any]:
        prompt = self.build_prompt_ids(question)
        open_ids = self.markers.reasoning_open
        ends_with_open = prompt[-len(open_ids) :] == open_ids if len(prompt) >= len(open_ids) else False
        if ends_with_open != self.prompt_contains_reasoning_open:
            raise RuntimeError(
                f"{self.name} prompt/open-marker mismatch: expected "
                f"prompt_contains_reasoning_open={self.prompt_contains_reasoning_open}, got {ends_with_open}"
            )
        if not self.markers.answer_boundary or not self.markers.terminal_ids:
            raise RuntimeError(f"{self.name} has incomplete boundary markers")
        return {
            "adapter": self.name,
            "prompt_token_count": len(prompt),
            "prompt_sha256": sha256_ints(prompt),
            "prompt_contains_reasoning_open": ends_with_open,
            "reasoning_open_ids": self.markers.reasoning_open,
            "answer_boundary_ids": self.markers.answer_boundary,
            "terminal_ids": self.markers.terminal_ids,
        }


class Qwen35Adapter(ReasoningAdapter):
    name = "qwen35"
    prompt_contains_reasoning_open = True

    def _markers(self) -> MarkerSet:
        eos = int(self.tokenizer.eos_token_id)
        pad = int(self.tokenizer.pad_token_id)
        return MarkerSet(
            reasoning_open=self._encode("<think>\n"),
            answer_boundary=self._encode("</think>\n\n"),
            terminal_ids=[eos],
            pad_token_id=pad,
        )

    def build_prompt_ids(self, question: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return self._encode(rendered)


class GptOssAdapter(ReasoningAdapter):
    name = "gpt_oss"
    prompt_contains_reasoning_open = False

    def _markers(self) -> MarkerSet:
        eos = int(self.tokenizer.eos_token_id)
        pad = int(self.tokenizer.pad_token_id)
        return MarkerSet(
            reasoning_open=self._encode("<|channel|>analysis<|message|>"),
            answer_boundary=self._encode(
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            ),
            terminal_ids=[eos],
            pad_token_id=pad,
        )

    def build_prompt_ids(self, question: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.profile.get("reasoning_effort", "medium"),
            current_date=self.profile.get("template_current_date", "2026-07-18"),
        )
        return self._encode(rendered)

class Ministral3ReasoningAdapter(ReasoningAdapter):
    name = "ministral3_reasoning"
    prompt_contains_reasoning_open = True
    reasoning_open_optional = True

    def __init__(self, processor_or_tokenizer: Any, profile: dict[str, Any]) -> None:
                                                                               
                                                                            
        self.processor = processor_or_tokenizer
        self.tokenizer = processor_or_tokenizer
        self.profile = profile
        self.markers = self._markers()

    def _markers(self) -> MarkerSet:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = self.tokenizer.convert_tokens_to_ids("</s>")
        eos = int(eos_id)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        pad = int(pad_id) if pad_id is not None else eos
        return MarkerSet(
            reasoning_open=self._encode("[THINK]"),
            answer_boundary=self._encode("[/THINK]"),
            terminal_ids=[eos],
            pad_token_id=pad,
        )

    def build_prompt_ids(self, question: str) -> list[int]:
        values = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            return_dict=False,
        )
        if isinstance(values, dict):
            values = values["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values] + list(self.markers.reasoning_open)

class Gemma4Adapter(ReasoningAdapter):
    name = "gemma4"
    prompt_contains_reasoning_open = False

    def _markers(self) -> MarkerSet:
        tokenizer = self.tokenizer
        terminal = int(tokenizer.convert_tokens_to_ids("<turn|>"))
        eos = int(tokenizer.eos_token_id)
        pad = int(tokenizer.pad_token_id)
        return MarkerSet(
            reasoning_open=self._encode("<|channel>thought\n"),
            answer_boundary=[int(tokenizer.convert_tokens_to_ids("<channel|>"))],
            terminal_ids=list(dict.fromkeys([terminal, eos])),
            pad_token_id=pad,
        )

    def build_prompt_ids(self, question: str) -> list[int]:
        messages = [
            {"role": "system", "content": self.profile["system_prompt"]},
            {"role": "user", "content": question},
        ]
        values = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(self.profile.get("enable_thinking", True)),
            return_tensors="pt",
        )
        if isinstance(values, dict):
            values = values["input_ids"]
        return [int(value) for value in values[0].tolist()]


def make_adapter(name: str, processor_or_tokenizer: Any, profile: dict[str, Any]) -> ReasoningAdapter:
    adapters = {
        "qwen35": Qwen35Adapter,
        "gpt_oss": GptOssAdapter,
        "ministral3_reasoning": Ministral3ReasoningAdapter,
        "gemma4": Gemma4Adapter,
    }
    try:
        return adapters[name](processor_or_tokenizer, profile)
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {name!r}") from exc
