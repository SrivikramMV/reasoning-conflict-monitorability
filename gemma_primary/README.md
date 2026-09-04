# Gemma primary collection

This folder contains the final code I used for my main Gemma 4 E2B collection
on an NVIDIA A100.

Gemma was the model I used throughout the early stages of the research, so
this was also the first model for which I ran the complete larger-scale
behavioural experiment. The collection includes the different trace-transfer
depths, the behavioural controls, the answer-only experiment, and the Gemma
results that I later used when developing the answer-boundary diagnostics.

`Gemma_E2B_Final_A100_Run.ipynb` is the Colab notebook I used to run the final
collection. It expects the corresponding Gemma experiment bundle, an A100
runtime, and Hugging Face access where required.
