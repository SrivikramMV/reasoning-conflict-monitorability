from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from nmt.activation_capture import CapturedRun, generated_span, generate_and_capture
from nmt.contrastive import nearest_runs, q_vs_qstar_similarity, similarity_by_layer
from nmt.exporting import lightweight_run_json, token_table
from nmt.logit_lens import top_tokens_grid, transcript_by_layer
from nmt.model_loader import ModelSettings, load_model_and_tokenizer
from nmt.phrase_reconstruction import reconstruct_phrase_summary
from nmt.prompting import PromptBundle, build_context


st.set_page_config(
    page_title="NMT Gemma Activation Translator",
    page_icon="NMT",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def cached_load_model(settings_dict: dict):
    settings = ModelSettings(**settings_dict)
    return load_model_and_tokenizer(settings)


def load_model_with_ui(settings_dict: dict, status_box):
    messages = []

    def progress(line: str):
        messages.append(line)
        status_box.code("\n".join(messages[-20:]))

    settings = ModelSettings(**settings_dict)
    return load_model_and_tokenizer(settings, progress_callback=progress)


def init_state():
    st.session_state.setdefault("runs", {})
    st.session_state.setdefault("current_run_label", None)


def sidebar_model():
    st.sidebar.header("Model")
    model_id = st.sidebar.text_input(
        "Model ID or local path",
        value="google/gemma-4-E2B-it",
        help="This is the Gemma 4 E2B checkpoint used in your experiment notebooks. If Hugging Face access is required, run huggingface-cli login first.",
    )
    device_mode = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    dtype = st.sidebar.selectbox("dtype", ["auto", "bfloat16", "float16", "float32"], index=0)
    load_in_4bit = st.sidebar.checkbox("Try 4-bit loading", value=False)
    trust_remote_code = st.sidebar.checkbox("trust_remote_code", value=False)
    local_files_only = st.sidebar.checkbox("local_files_only", value=False)
    settings = {
        "model_id": model_id,
        "device_mode": device_mode,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "load_in_4bit": load_in_4bit,
        "local_files_only": local_files_only,
    }
    if st.sidebar.button("Load model", type="primary"):
        st.session_state["model_loaded"] = False
        st.sidebar.info("Loading has started. Watch the status panel in the main page.")
        st.subheader("Model Load Status")
        status_box = st.empty()
        with st.spinner("Loading Gemma. This may download several GB on the first run..."):
            model, tokenizer = load_model_with_ui(settings, status_box)
                                                                           
        cached_load_model.clear()
        st.session_state["loaded_model_settings"] = settings
        st.session_state["loaded_model_object"] = (model, tokenizer)
        st.session_state["model_loaded"] = True
        st.success("Model loaded.")
    return settings


def require_model(settings):
    if not st.session_state.get("model_loaded"):
        st.info("Load a model from the sidebar first.")
        st.stop()
    if st.session_state.get("loaded_model_settings") == settings and "loaded_model_object" in st.session_state:
        return st.session_state["loaded_model_object"]
    return cached_load_model(settings)


def run_input_panel(model, tokenizer):
    st.subheader("Prompt and CoT Injection")
    col1, col2 = st.columns(2)
    with col1:
        label = st.text_input("Run label", value="injected_run")
        system_prompt = st.text_input("System prompt", value="You are a helpful assistant.")
        prompt = st.text_area(
            "Original prompt Q",
            height=160,
            value="Evaluate: 18 + 6 * 4 - 9.",
        )
        final_instruction = st.text_area(
            "Optional final-answer instruction",
            height=70,
            value="",
            placeholder="Example: Give the final answer only.",
        )
        use_chat_template = st.checkbox("Use tokenizer chat template", value=True)
        enable_thinking = st.checkbox("Enable Gemma thinking template", value=True)
    with col2:
        injected_cot = st.text_area(
            "Optional injected / prefilled CoT prefix",
            height=230,
            value=(
                "I need to evaluate the expression using order of operations.\n"
                "First compute the multiplication: 7 * 4 = 28.\n"
                "Then 18 + 28 - 9 = 37.\n"
            ),
        )
        injection_is_raw = st.checkbox(
            "Injected CoT is already raw Gemma text",
            value=False,
            help="Leave unchecked for normal CoT text. NMT will add <|channel>thought automatically, matching your notebooks.",
        )
    raw_override = st.text_area(
        "Raw full context override",
        height=90,
        value="",
        help="If supplied, this exact string is sent to the model. Useful if you want to paste exact Gemma control tokens.",
    )
    context = build_context(
        PromptBundle(
            prompt=prompt,
            injected_cot=injected_cot,
            final_instruction=final_instruction,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            injection_is_raw=injection_is_raw,
            raw_override=raw_override,
        ),
        tokenizer=tokenizer,
    )
    with st.expander("Context sent to model", expanded=False):
        st.code(context)

    st.subheader("Generation Settings")
    c1, c2, c3, c4 = st.columns(4)
    max_new_tokens = c1.number_input("max_new_tokens", min_value=1, max_value=2048, value=128, step=16)
    temperature = c2.number_input("temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
    top_p = c3.number_input("top_p", min_value=0.05, max_value=1.0, value=1.0, step=0.05)
    seed = c4.number_input("seed", min_value=0, max_value=999999, value=0, step=1)

    if st.button("Generate and capture activations", type="primary"):
        with st.spinner("Generating and running activation capture..."):
            run = generate_and_capture(
                model,
                tokenizer,
                context,
                label=label,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=int(seed),
            )
        st.session_state["runs"][label] = run
        st.session_state["current_run_label"] = label
        st.success(f"Captured run: {label}")


def current_run() -> CapturedRun | None:
    label = st.session_state.get("current_run_label")
    if not label:
        return None
    return st.session_state["runs"].get(label)


def run_selector(key_prefix: str = "run_selector") -> CapturedRun | None:
    labels = list(st.session_state["runs"].keys())
    if not labels:
        st.info("No captured runs yet.")
        return None
    current = st.session_state.get("current_run_label") or labels[-1]
    chosen = st.selectbox(
        "Captured run",
        labels,
        index=labels.index(current) if current in labels else 0,
        key=f"{key_prefix}_captured_run",
    )
    st.session_state["current_run_label"] = chosen
    return st.session_state["runs"][chosen]


def token_position_controls(run: CapturedRun, default_generated=True, key_prefix: str = "token"):
    if default_generated and run.input_token_count < run.seq_len:
        default_pos = run.input_token_count
    else:
        default_pos = max(0, run.seq_len - 1)
    pos = st.slider(
        "Token position",
        min_value=0,
        max_value=max(0, run.seq_len - 1),
        value=min(default_pos, max(0, run.seq_len - 1)),
        key=f"{key_prefix}_token_position",
    )
    st.caption(f"Selected token: `{run.tokens[pos]}` | region: {'prefilled context' if pos < run.input_token_count else 'generated'}")
    return pos


def generated_span_controls(run: CapturedRun, key_prefix: str = "span"):
    default_start, default_end = generated_span(run)
    if default_start >= default_end:
        default_start = max(0, run.seq_len - 16)
        default_end = run.seq_len
    start, end = st.slider(
        "Span for phrase/similarity analysis",
        min_value=0,
        max_value=max(1, run.seq_len),
        value=(max(0, default_start), max(1, default_end)),
        key=f"{key_prefix}_generated_span",
    )
    if start == end:
        end = min(run.seq_len, start + 1)
    return int(start), int(end)


def render_generation_tab(settings):
    model, tokenizer = require_model(settings)
    run_input_panel(model, tokenizer)

    run = current_run()
    if run is None:
        return

    st.divider()
    st.subheader("Generated Output")
    st.code(run.generated_text)

    with st.expander("Token table", expanded=False):
        st.dataframe(token_table(run), use_container_width=True)

    st.subheader("Quick Translation Summary")
    start, end = generated_span_controls(run, key_prefix="generate_summary")
    answer_a = st.text_input("Optional correct answer A", value="", key="generate_summary_answer_a")
    answer_astar = st.text_input("Optional counterfactual answer A*", value="", key="generate_summary_answer_astar")
    try:
        summaries = reconstruct_phrase_summary(
            model,
            tokenizer,
            run,
            start,
            end,
            answer_a=answer_a,
            answer_astar=answer_astar,
        )
        for summary in summaries:
            with st.container(border=True):
                st.markdown(f"**{summary.title}**")
                st.write(summary.text)
                if summary.evidence:
                    st.caption("Evidence: " + "; ".join(str(x) for x in summary.evidence[:10]))
    except Exception as exc:
        st.error(f"Phrase reconstruction failed: {type(exc).__name__}: {exc}")
        st.info("The run was still captured. Try a shorter span or inspect the Logit Lens tab.")


def render_logit_lens_tab(settings):
    model, tokenizer = require_model(settings)
    run = run_selector(key_prefix="logit_lens")
    if run is None:
        return
    st.subheader("Layer-to-Words Logit Lens")
    st.write(
        "This view projects each layer's hidden state through the model's vocabulary head. "
        "It shows what tokens are linearly decodable from intermediate representations."
    )
    pos = token_position_controls(run, key_prefix="logit_lens")
    top_k = st.slider("Top-k tokens per layer", 3, 30, 10, key="logit_lens_top_k")
    try:
        grid = pd.DataFrame(top_tokens_grid(model, tokenizer, run, pos, top_k=top_k))
        st.dataframe(grid, use_container_width=True)
    except Exception as exc:
        st.error(f"Logit lens failed: {type(exc).__name__}: {exc}")

    st.subheader("Approximate Layer Transcript")
    start, end = generated_span_controls(run, key_prefix="layer_transcript")
    stride = st.slider("Layer stride", 1, 8, 2, key="layer_transcript_stride")
    try:
        transcripts = pd.DataFrame(transcript_by_layer(model, tokenizer, run, start, end, stride=stride))
        st.dataframe(transcripts, use_container_width=True)
    except Exception as exc:
        st.error(f"Layer transcript failed: {type(exc).__name__}: {exc}")


def render_compare_tab(settings):
    require_model(settings)
    labels = list(st.session_state["runs"].keys())
    if len(labels) < 2:
        st.info("Capture at least two runs to compare them.")
        return

    st.subheader("Run Similarity")
    c1, c2 = st.columns(2)
    cur_label = c1.selectbox("Current run", labels, index=len(labels) - 1, key="compare_current_run")
    ref_label = c2.selectbox("Reference run", labels, index=0, key="compare_reference_run")
    cur = st.session_state["runs"][cur_label]
    ref = st.session_state["runs"][ref_label]
    cur_span = generated_span_controls(cur, key_prefix="compare_current")
    st.caption("Reference span uses generated tokens by default.")
    ref_span = generated_span(ref)
    rows = pd.DataFrame(similarity_by_layer(cur, ref, cur_span, ref_span))
    fig = px.line(rows, x="layer", y="similarity", markers=True, title=f"{cur_label} vs {ref_label}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rows, use_container_width=True)

    st.subheader("Q vs Q* Contrast")
    c1, c2, c3 = st.columns(3)
    target_label = c1.selectbox("Target run", labels, index=labels.index(cur_label), key="compare_target_run")
    q_label = c2.selectbox("Clean Q run", labels, index=0, key="compare_clean_q_run")
    qstar_label = c3.selectbox("Clean Q* run", labels, index=min(1, len(labels) - 1), key="compare_clean_qstar_run")
    target = st.session_state["runs"][target_label]
    q_run = st.session_state["runs"][q_label]
    qstar_run = st.session_state["runs"][qstar_label]
    target_span = generated_span(target)
    contrast = pd.DataFrame(
        q_vs_qstar_similarity(
            target,
            q_run,
            qstar_run,
            target_span,
            generated_span(q_run),
            generated_span(qstar_run),
        )
    )
    fig2 = px.line(
        contrast,
        x="layer",
        y=["similarity_to_Q", "similarity_to_Qstar"],
        markers=True,
        title=f"{target_label}: similarity to Q vs Q*",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(contrast, use_container_width=True)

    st.subheader("Nearest Previously Captured Runs")
    layer = st.slider(
        "Layer for nearest-neighbour comparison",
        0,
        target.layer_count - 1,
        target.layer_count - 1,
        key="nearest_layer",
    )
    nn = pd.DataFrame(nearest_runs(target, st.session_state["runs"], target_span, layer=layer))
    st.dataframe(nn, use_container_width=True)


def render_export_tab():
    labels = list(st.session_state["runs"].keys())
    if not labels:
        st.info("No captured runs yet.")
        return
    run = run_selector(key_prefix="export")
    if run is None:
        return
    st.subheader("Export")
    payload = lightweight_run_json(run)
    st.download_button(
        "Download lightweight run JSON",
        data=json.dumps(payload, indent=2),
        file_name=f"{run.label}_nmt_run.json",
        mime="application/json",
    )
    st.download_button(
        "Download token table CSV",
        data=token_table(run).to_csv(index=False),
        file_name=f"{run.label}_tokens.csv",
        mime="text/csv",
    )
    st.warning(
        "Hidden states are kept in memory for interactive analysis but are not included in the lightweight export. "
        "Full activation export can become very large and should be added only for selected cases."
    )


def main():
    init_state()
    st.title("NMT: Gemma Activation Translator")
    st.caption(
        "A local logit-lens, phrase-reconstruction, and contrastive activation dashboard for CoT faithfulness probes."
    )
    settings = sidebar_model()

    tab_generate, tab_lens, tab_compare, tab_export, tab_notes = st.tabs(
        ["Generate", "Logit Lens", "Compare Runs", "Export", "Interpretation Notes"]
    )
    with tab_generate:
        render_generation_tab(settings)
    with tab_lens:
        render_logit_lens_tab(settings)
    with tab_compare:
        render_compare_tab(settings)
    with tab_export:
        render_export_tab()
    with tab_notes:
        st.markdown(
            """
### How to interpret the outputs

The logit lens gives real vocabulary tokens decoded from intermediate hidden states, but those tokens are not literal hidden thoughts. A careful interpretation is:

> The token `40` is linearly decodable from late-layer states before the final answer.

not:

> The model consciously thought `40`.

The most useful workflow for the dissertation project is contrastive:

1. Run clean `Q`.
2. Run clean `Q*`.
3. Run injected `Q + CoT(Q*)`.
4. Compare the injected run to both clean references across layers.
5. Inspect whether `A`, `A*`, `Q`, or `Q*` terms become decodable before the final answer.

If an answer-stage bypass case is visibly still on `Q*` at the boundary but late-layer states look closer to `Q`, that is evidence of internal prompt re-anchoring not verbalised in the CoT. If a FOLLOW case remains closer to `Q*`, that supports trace-state inheritance.
"""
        )


if __name__ == "__main__":
    main()
