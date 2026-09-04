# Additional experiments

This folder contains experiments that I worked on while trying to understand
the results from the main behavioural study and decide where to take the
research next. Not all of these eventually became part of my final
dissertation, but they were useful in getting me from the initial behavioural
taxonomy to the final monitorability experiments.

The folders roughly cover the following parts of that process:

- `early_counterfactual_benchmark` and `early_taxonomy_analysis` contain my
  first scaled version of the behavioural experiment and the analysis that
  followed from it.
- `two_stage_trace_transfer` contains the later two-stage version of the
  trace-transfer experiment.
- `linear_answer_boundary_probe` and `linear_cot_source_ablation` contain some
  of my earlier attempts at understanding whether the forthcoming branch could
  already be recovered at the answer boundary and what information it depended
  on.
- `hidden_state_pilots` and `neural_monitoring_prototype` contain some of my
  earlier experiments using the models' internal representations.
- `activation_patching` contains the activation-patching experiments I
  explored while trying to understand the internal cause of the different
  behavioural pathways. I ultimately decided not to use these as part of the
  main dissertation results.
- `diagnostic_smoke_tests` contains smaller tests I used while checking the
  setup before running the final diagnostic collection.
- `grounding_and_transparency` contains experiments where I changed how the
  reasoning trace was presented to the model and tested whether the model
  could accurately report the reasoning it had been given.
- `ministral_reasoning_extension` contains my attempted extension to a
  Ministral reasoning setup. I eventually excluded this model because its
  prompted reasoning format created problems with trace completion and made
  the setup difficult to compare fairly with the three models I retained.

These folders should not be treated as one pipeline that needs to be run from
beginning to end. They are different experiments I tried at different stages
of the research. The final collection code is available separately at the
repository root and inside `gemma_primary`, `cross_model_primary`, and
`diagnostics`.
