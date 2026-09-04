# Monitoring Conflict Resolution in Reasoning Language Models

This repository contains the source code I used throughout my research on
**Monitoring Conflict Resolution in Reasoning Language Models**.

I've kept the code from the beginning of my exploratory stages, including
experiments that did not ultimately make it into my final dissertation, the
final collection pipelines, and a curated set of figures that I generated at
different points throughout the research.

## How my research developed

I started this project by working only with Gemma 4 E2B. At this point, I
was mainly trying to understand what actually happens when I give the model
one question while placing reasoning from a slightly different counterfactual
question inside its chain of thought.

A lot of my work at the beginning was therefore exploratory. I built an
interactive notebook where I could enter the original question, give the model
a conflicting reasoning trace, control how much of that trace was transferred,
and then look at what the model did afterwards. I intentionally stayed with
one model while doing this because I first wanted to understand the behaviour
properly before scaling the experiments further.

This is where the four behaviours used throughout my final research started
to emerge. After manually going through the outputs from Gemma, I noticed that
the model was resolving the conflict in a few noticeably different ways. I
eventually separated these into the four pathways used in my taxonomy:

- **CoT-stage correction**, where the model corrects the transferred reasoning
  before reaching the final answer.
- **Follow**, where the model continues with the counterfactual reasoning and
  gives the corresponding counterfactual answer.
- **Answer-stage bypass**, where the chain of thought supports the
  counterfactual problem, but the model suddenly returns to the original
  question during its final response.
- **Final-stage transparent correction**, where the model returns to the
  original question but actually makes this correction visible in its final
  response.

Once I had a better understanding of these behaviours, I started scaling the
experiments. I moved from the interactive notebook into a two-stage pipeline.
Stage A generated and checked the source reasoning traces, and Stage B took
those traces and transferred different amounts of them into the paired
questions. One of my earlier larger runs contained 300 source prompts and
600 interventions, which gave me enough results to start looking at the
taxonomy beyond individual examples and generate some of the first larger
figures in the project.

From there, the research became much more exploratory again. I tried quite a
few different ideas while trying to understand why the models behaved the way
they did. These included answer-only controls, linear probes at the answer
boundary, source ablations, hidden-state capture, activation patching,
repeated sampling, candidate scoring, masking experiments, different ways of
showing or hiding the trace, and grounding requests.

A number of these experiments did not end up becoming part of my final
dissertation. I've still kept the latest useful versions of them in this
repository because they were part of how I arrived at the final direction of
the research. In some cases an experiment gave me an interesting result but
didn't fit the final research question well enough. In other cases it simply
didn't give me enough useful information to justify taking it further.

I also tried different model setups during this process. Gemma was the model
I used throughout the initial exploration, and I later expanded the final
experiments to GPT-OSS 20B and Qwen 3.5 9B. I also prepared and tested a
Ministral reasoning setup at one point, but its prompted reasoning format
created problems with comparability and trace completion, so I eventually
decided not to include it as one of the three final models. The latest version
of that work is still available under `additional_experiments`.

One of the biggest changes in direction happened after I started looking more
closely at the full-trace results. I had cases where the visible chain of
thought looked extremely similar and supported the same counterfactual
problem, but the models could still behave differently immediately after the
answer boundary. I spent a considerable amount of time trying to find
something in the visible reasoning that could explain this. I looked at
things such as trace length, similarity, fractions in the solution,
verification steps, how the reasoning ended, and even individual wording
differences between similar traces.

I wasn't able to find anything from these tests that reliably explained why
one case would follow the transferred reasoning while another would bypass it.
That eventually changed the direction of the project. Instead of only asking
how faithful the visible chain of thought was, I started asking whether the
model's forthcoming behaviour could still be monitored when the visible
reasoning itself was no longer enough to reliably tell me what would happen
next.

This led to the final set of diagnostic experiments used in the dissertation.
I first used repeated continuation sampling to check whether the branch I was
observing was actually a stable nearby preference rather than one arbitrary
generation. I then used candidate-conditioned scoring at the answer boundary
to see whether the model already favoured one of the two branches before its
natural response began. I also explored whether similar information could be
recovered from the hidden states, and whether directly asking the model about
the supplied reasoning could provide useful information without changing the
behaviour I was trying to observe.

The final results showed that the visible chain of thought did not always
contain everything that could be learned about what the model was likely to
do next. Candidate-conditioned scoring was particularly informative for some
of the models, while the hidden-state results were considerably more
model-dependent. The three models also behaved very differently in how and
when they resolved the same controlled conflict.

## Repository layout

The repository contains both the final experimental code and the earlier work
that led up to it.

- `src/` and `counterfactual_monitorability_final_completion_v3_colab.ipynb`
  contain the final completion and recovery runner at the root of the
  repository.
- `gemma_primary/` contains the final Gemma A100 collection code.
- `cross_model_primary/` contains the final cross-model behavioural collection
  code for GPT-OSS and Qwen, together with the remaining Gemma control
  collection.
- `diagnostics/` contains the final answer-boundary diagnostic experiments for
  all three models.
- `notebooks/` contains much of my earlier notebook-based exploration,
  including the work that eventually led to the behavioural taxonomy and the
  first scaled experiments.
- `additional_experiments/` contains experiments I explored during the project
  that were useful to the development of the research but did not become part
  of the final experimental pipeline.
- `research_figures/` contains a selection of figures I generated throughout
  the project, including early Gemma analyses, answer-boundary experiments,
  the cross-model expansion, and the final dissertation figures.

The commit history also contains earlier versions of the code, fixes, and
changes I made as the research developed. I kept these in the history rather
than keeping every obsolete version of the runtime code in the latest tree.

## Running the final experiments

I ran the final experiments primarily through Google Colab using an NVIDIA
A100. The notebooks use Hugging Face model access and save checkpointed outputs
to Google Drive so that longer runs could be recovered if the runtime was
interrupted.

The easiest way to reproduce the setup I used is:

1. Open the relevant Colab notebook.
2. Select an A100 GPU runtime.
3. Add `HF_TOKEN` to the Colab secrets if the selected model requires Hugging
   Face authentication.
4. Upload the corresponding experiment bundle using the filename expected by
   the notebook.
5. Run the notebook from top to bottom. It will install the required
   dependencies, check the runtime, start the experiment, and save checkpoints
   and results to Google Drive.

The main entry points I used for the final experiments are:

- `gemma_primary/Gemma_E2B_Final_A100_Run.ipynb` for the main Gemma experiments.
- `cross_model_primary/counterfactual_monitorability_cross_model_v2_colab.ipynb`
  for the GPT-OSS and Qwen experiments.
- `counterfactual_monitorability_final_completion_v3_colab.ipynb` for the
  focused completion and recovery runs.
- `diagnostics/counterfactual_monitorability_final_diagnostics_v2_colab.ipynb`
  for the final three-model diagnostic experiments.

If the original bundle structure is already available locally, the completion
runner can also be called directly:

```bash
python src/run_final_completion.py --bundle-root /path/to/bundle --output-root /path/to/output
```
