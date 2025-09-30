# VAL-Bench
A diverse benchmark for systematic analysis of a how reliably language models embody human values.

## Abstract
Large language models (LLMs) are increasingly used for tasks where outputs shape human decisions, so it is critical to test whether their responses reflect consistent human values. Existing benchmarks mostly track refusals or predefined safety violations, but these only check rule compliance and do not reveal whether a model upholds a coherent value system when facing controversial real-world issues. We introduce the Value ALignment Benchmark (VAL-Bench), which evaluates whether models maintain a stable value stance across paired prompts that frame opposing sides of public debates. VAL-Bench consists of 115K such pairs from Wikipedia’s controversial sections. A well-aligned model should express similar underlying views regardless of framing, which we measure using an LLM-as-judge to score agreement or divergence between paired responses. Applied across leading open- and closed-source models, the benchmark reveals large variation in alignment and highlights trade-offs between safety strategies (e.g., refusals) and more expressive value systems. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic comparison of how reliably LLMs embody human values.

![Alt text](VAL-Bench.png)

## Datasets

All datasets are hosted on HuggingFace.

[VAL-Bench](https://huggingface.co/datasets/val-bench/VAL-Bench)

[VAL-Bench-test-w-values](https://huggingface.co/datasets/val-bench/VAL-Bench-test-w-values)

### Calibration Datasets

As described in the paper, these datasets have responses that can be used to calibrate the evaluator.

[unaligned-0](https://huggingface.co/datasets/val-bench/calibration-unaligned-0)

[aligned-100](https://huggingface.co/datasets/val-bench/calibration-aligned-100)

[refusal-100](https://huggingface.co/datasets/val-bench/calibration-refusal-100)

[one-refusal-50](https://huggingface.co/datasets/val-bench/calibration-one-refusal-50)


## Running the Evaluator

_Coming soon_
