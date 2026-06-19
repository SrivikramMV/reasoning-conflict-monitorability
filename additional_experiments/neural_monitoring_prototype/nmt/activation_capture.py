from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .model_loader import model_device


@dataclass
class CapturedRun:
    label: str
    context_text: str
    generated_text: str
    full_text: str
    input_token_count: int
    token_ids: List[int]
    tokens: List[str]
    hidden_states_cpu: List[torch.Tensor]
    logits_cpu: torch.Tensor
    metadata: Dict[str, str]

    @property
    def layer_count(self) -> int:
        return len(self.hidden_states_cpu)

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)


def _token_strings(tokenizer, token_ids: List[int]) -> List[str]:
    out = []
    for tid in token_ids:
        text = tokenizer.decode([tid], skip_special_tokens=False)
        out.append(text.replace("\n", "\\n"))
    return out


@torch.inference_mode()
def generate_and_capture(
    model,
    tokenizer,
    context_text: str,
    label: str = "run",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> CapturedRun:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = model_device(model)
    encoded = tokenizer(context_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_len = int(encoded["input_ids"].shape[-1])

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    generated = model.generate(**encoded, **generation_kwargs)
    full_ids = generated.sequences[0]
    generated_ids = full_ids[input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

                                                                        
                                                                                
                                                  
    out = model(
        input_ids=full_ids.unsqueeze(0).to(device),
        attention_mask=torch.ones_like(full_ids, device=device).unsqueeze(0),
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states_cpu = [h[0].detach().float().cpu() for h in out.hidden_states]
    logits_cpu = out.logits[0].detach().float().cpu()
    ids_list = [int(x) for x in full_ids.detach().cpu().tolist()]

    return CapturedRun(
        label=label,
        context_text=context_text,
        generated_text=generated_text,
        full_text=full_text,
        input_token_count=input_len,
        token_ids=ids_list,
        tokens=_token_strings(tokenizer, ids_list),
        hidden_states_cpu=hidden_states_cpu,
        logits_cpu=logits_cpu,
        metadata={
            "max_new_tokens": str(max_new_tokens),
            "temperature": str(temperature),
            "top_p": str(top_p),
        },
    )


def pooled_hidden(
    run: CapturedRun,
    layer: int,
    start: int,
    end: int,
) -> torch.Tensor:
    start = max(0, min(start, run.seq_len - 1))
    end = max(start + 1, min(end, run.seq_len))
    return run.hidden_states_cpu[layer][start:end].mean(dim=0)


def generated_span(run: CapturedRun) -> tuple[int, int]:
    return run.input_token_count, run.seq_len
