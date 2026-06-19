from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptBundle:
    prompt: str
    injected_cot: str = ""
    final_instruction: str = ""
    use_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant."
    enable_thinking: bool = True
    injection_is_raw: bool = False
    raw_override: str = ""


def build_context(bundle: PromptBundle, tokenizer=None) -> str:







    if bundle.raw_override.strip():
        return bundle.raw_override

    user_content = bundle.prompt.strip()
    if bundle.final_instruction.strip():
        user_content = f"{user_content}\n\n{bundle.final_instruction.strip()}"

    if bundle.use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            base = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": bundle.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bundle.enable_thinking,
            )
        except TypeError:
            base = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": bundle.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            base = f"{bundle.system_prompt}\n\n{user_content}\n\n"
    else:
        base = f"{bundle.system_prompt}\n\n{user_content}\n\n"

    if bundle.injected_cot.strip():
        injected = bundle.injected_cot.strip()
        if not bundle.injection_is_raw and not injected.lstrip().startswith("<|channel>thought"):
            injected = "<|channel>thought\n" + injected
        return base + injected
    return base
