# Cross-model primary collection

After completing the main Gemma experiments, I wanted to see how much of the
behaviour I was observing was specific to Gemma and how much would carry over
to other reasoning models. This folder contains the final behavioural
collection I used to extend the experiment to GPT-OSS 20B and Qwen 3.5 9B,
along with the remaining Gemma control collection.

The Colab notebook sets up the runtime, loads the corresponding experiment
bundle, and then starts `src/run_all.py`. The individual model runner is also
available in `src/run_replication_model.py` for running one of the model
collections separately.
