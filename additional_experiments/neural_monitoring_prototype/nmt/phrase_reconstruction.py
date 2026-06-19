from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import string
from typing import Iterable, List, Optional

from .activation_capture import CapturedRun
from .logit_lens import top_tokens_for_state


ANSWER_WORDS = {
    "answer",
    "final",
    "equals",
    "therefore",
    "result",
    "value",
    "solution",
}

CORRECTION_WORDS = {
    "wait",
    "check",
    "recheck",
    "reread",
    "read",
    "mistake",
    "actually",
    "correct",
    "prompt",
    "question",
}

TRACE_WORDS = {
    "scratchpad",
    "previous",
    "reasoning",
    "above",
    "calculation",
    "computed",
}


@dataclass
class PhraseSummary:
    title: str
    text: str
    evidence: List[str]


def clean_token(token: str) -> str:
    token = token.replace("▁", " ").replace("\\n", " ").strip()
    token = token.replace("<bos>", "").replace("<eos>", "").strip()
    return token


def readable_token(token: str) -> bool:

    if not token:
        return False
    if len(token) > 28:
        return False
    if any(marker in token for marker in ["<", ">", "�", "\ufffd"]):
        return False
    stripped = token.strip()
    if not stripped:
        return False
    if re.fullmatch(r"[-+*/=().,:;$]+", stripped):
        return True
    if stripped in string.punctuation:
        return False
    if re.fullmatch(r"\d+([./]\d+)?", stripped):
        return True
    ascii_chars = sum(1 for ch in stripped if ord(ch) < 128)
    ascii_ratio = ascii_chars / max(1, len(stripped))
    if ascii_ratio < 0.85:
        return False
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
                                                                           
                                                                                  
    if re.search(r"[a-z][A-Z][a-z]", stripped):
        return False
    return True


def collect_decoded_vocabulary(
    model,
    tokenizer,
    run: CapturedRun,
    start: int,
    end: int,
    layers: Iterable[int],
    top_k: int = 8,
) -> Counter:
    counts: Counter = Counter()
    for layer in layers:
        if layer < 0 or layer >= run.layer_count:
            continue
        for pos in range(max(0, start), min(end, run.seq_len)):
            toks = top_tokens_for_state(model, tokenizer, run.hidden_states_cpu[layer][pos], top_k=top_k)
            for rank, tok in enumerate(toks):
                cleaned = clean_token(tok.token)
                if readable_token(cleaned):
                    counts[cleaned] += max(1, top_k - rank)
    return counts


def _contains_any(words: Iterable[str], vocab: set[str]) -> List[str]:
    hits = []
    for word in words:
        lower = word.lower().strip(" .,:;!?()[]{}")
        if lower in vocab:
            hits.append(word)
    return hits


def reconstruct_phrase_summary(
    model,
    tokenizer,
    run: CapturedRun,
    start: int,
    end: int,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    top_k: int = 8,
    answer_a: str = "",
    answer_astar: str = "",
) -> List[PhraseSummary]:
    if layer_start is None:
        layer_start = max(0, int(run.layer_count * 0.55))
    if layer_end is None:
        layer_end = run.layer_count
    layers = range(max(0, layer_start), min(run.layer_count, layer_end))
    vocab = collect_decoded_vocabulary(model, tokenizer, run, start, end, layers, top_k=top_k)
    common = [w for w, _ in vocab.most_common(30)]

    summaries: List[PhraseSummary] = []
    if len(common) >= 3:
        summaries.append(
            PhraseSummary(
                title="Dominant decoded vocabulary",
                text="Late-layer states most often decode to words/tokens such as: "
                + ", ".join(common[:15])
                + ".",
                evidence=[f"{w}: {vocab[w]}" for w in common[:12]],
            )
        )

    answer_hits = _contains_any(common, ANSWER_WORDS)
    correction_hits = _contains_any(common, CORRECTION_WORDS)
    trace_hits = _contains_any(common, TRACE_WORDS)

    if answer_hits:
        summaries.append(
            PhraseSummary(
                title="Answer-forming language",
                text=(
                    "The selected activations contain answer-forming vocabulary, suggesting that "
                    "the model may already be preparing an answer-like continuation in this span."
                ),
                evidence=answer_hits,
            )
        )

    if correction_hits:
        summaries.append(
            PhraseSummary(
                title="Re-read or correction language",
                text=(
                    "The selected activations include words associated with checking or correction. "
                    "This can be a useful signal in final-stage transparent-correction cases, although "
                    "it should be treated as suggestive rather than conclusive."
                ),
                evidence=correction_hits,
            )
        )

    if trace_hits:
        summaries.append(
            PhraseSummary(
                title="Trace-reference language",
                text=(
                    "The decoded vocabulary contains words that may refer to prior reasoning or a "
                    "scratchpad. This is worth inspecting manually because explicit trace reference is "
                    "central to transparent correction."
                ),
                evidence=trace_hits,
            )
        )

    answer_evidence = []
    for label, answer in [("A", answer_a), ("A*", answer_astar)]:
        if answer.strip():
            for piece in answer.replace(",", " ").split():
                if piece and piece in vocab:
                    answer_evidence.append(f"{label} token {piece!r}: {vocab[piece]}")
    if answer_evidence:
        summaries.append(
            PhraseSummary(
                title="Answer-token evidence",
                text=(
                    "Some supplied answer tokens appear in the decoded layer vocabulary. "
                    "This can indicate early answer pressure before the visible final answer."
                ),
                evidence=answer_evidence,
            )
        )

    if not summaries:
        summaries.append(
            PhraseSummary(
                title="No stable readable phrase reconstruction",
                text=(
                    "The selected span did not produce a stable readable phrase summary after filtering "
                    "raw tokenizer artefacts. Try a shorter span near the answer boundary, inspect the "
                    "Logit Lens table directly, or compare the run against clean Q and Q* references."
                ),
                evidence=[],
            )
        )
    return summaries
