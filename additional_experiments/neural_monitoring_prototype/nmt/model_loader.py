from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


LOAD_LOG_PATH = Path(__file__).resolve().parents[1] / "model_load_status.log"


def _log(message: str, progress_callback=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    try:
        LOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOAD_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if progress_callback is not None:
        progress_callback(line)


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "google/gemma-4-E2B-it"
    device_mode: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    local_files_only: bool = False


class ProcessorTokenizerAdapter:







    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        try:
            return self.processor(text=text, return_tensors=return_tensors, **kwargs)
        except TypeError:
            return self.tokenizer(text, return_tensors=return_tensors, **kwargs)

    def decode(self, *args, **kwargs):
        if hasattr(self.processor, "decode"):
            return self.processor.decode(*args, **kwargs)
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(*args, **kwargs)
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def resolve_dtype(dtype: str):
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_device_map(device_mode: str):
    if device_mode == "auto":
        return "auto"
    if device_mode == "cpu":
        return {"": "cpu"}
    if device_mode == "cuda":
        return {"": "cuda"}
    raise ValueError(f"Unsupported device mode: {device_mode}")


def load_model_and_tokenizer(settings: ModelSettings, progress_callback=None):






    _log(f"Starting model load for {settings.model_id}", progress_callback)
    _log("Loading processor/tokenizer metadata from Hugging Face cache or hub...", progress_callback)
    try:
        processor = AutoProcessor.from_pretrained(
            settings.model_id,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
        )
        tokenizer = ProcessorTokenizerAdapter(processor)
        _log(f"Processor loaded: {type(processor).__name__}", progress_callback)
    except Exception:
        _log("AutoProcessor failed; falling back to AutoTokenizer...", progress_callback)
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=settings.local_files_only,
        )
        _log(f"Tokenizer loaded: {type(tokenizer).__name__}", progress_callback)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        _log("Tokenizer pad token was missing; using EOS as pad token.", progress_callback)

    dtype = resolve_dtype(settings.dtype)
    device_map = resolve_device_map(settings.device_mode)
    _log(f"Resolved dtype={dtype}; device_map={device_map}", progress_callback)
    kwargs = {
        "trust_remote_code": settings.trust_remote_code,
        "local_files_only": settings.local_files_only,
        "device_map": device_map,
        "torch_dtype": dtype,
        "output_hidden_states": True,
    }

    if settings.load_in_4bit:
        _log("4-bit loading requested; preparing BitsAndBytesConfig...", progress_callback)
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs.pop("torch_dtype", None)
        except Exception as exc:                                               
            raise RuntimeError(
                "4-bit loading requested, but bitsandbytes/transformers quantisation "
                f"support is unavailable or broken on this machine: {exc}"
            ) from exc

    _log(
        "Loading model weights. If the model is not cached, this is the stage that downloads several GB.",
        progress_callback,
    )
    model = AutoModelForCausalLM.from_pretrained(settings.model_id, **kwargs)
    _log("Model weights loaded; setting eval mode...", progress_callback)
    model.eval()
    try:
        device = next(model.parameters()).device
        _log(f"Model ready on device: {device}", progress_callback)
    except Exception:
        _log("Model ready; could not inspect first parameter device.", progress_callback)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        _log(f"CUDA memory allocated={allocated:.2f}GB reserved={reserved:.2f}GB", progress_callback)
    _log("Finished model load.", progress_callback)
    return model, tokenizer


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
