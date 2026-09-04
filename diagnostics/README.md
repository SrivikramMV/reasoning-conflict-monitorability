# Standardised diagnostics

This folder contains the final diagnostic experiments I ran after the
behavioural results raised the question of whether the models' forthcoming
responses could still be monitored beyond the visible chain of thought.

I used a shared cohort of 29 question pairs across the three final models so
that the different diagnostics could be compared on the same set of cases.
The experiments here include candidate-conditioned scoring at the answer
boundary, repeated continuations at two temperatures, hidden-state capture,
and source-masking checks.

The Colab notebook checks the runtime, installs the required dependencies, and
starts `src/run_all_standardized_diagnostics.py`. I also added quality checks
to the collection code before finalising the outputs so that incomplete or
invalid runs could be identified before they were used for the later analysis.
