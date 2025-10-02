# VAL-Bench
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/val-bench/VAL-Bench)

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

We use [metaflow](https://metaflow.org/) to structure execution.

### One time setup

Install metaflow and other "global" dependencies.
```bash
pip install -r requirements.txt
cp .env.template .env
```

Add secrets to `.env` file.

### Generating responses from candidate model

If using OpenAI, Claude or Bedrock's APIs:

```bash
python val_bench_responses_flow.py --environment=pypi run --model_id="gpt-4.1-nano" --output_dataset_path="local_results/responses/val_responses"
```

This will generate a huggingface dataset at the path `local_results/responses/val_responses-gpt-4-1-nano`

If using another OpenAI compatible API (like a vllm server) with a custom URL

```bash
python val_bench_responses_flow.py --environment=pypi run --model_id="meta-llama/Llama-3.2-1B-Instruct" --inference_url="http://0.0.0.0:8000/v1" --output_dataset_path="local_results/responses/val_responses"
```

Run `python val_bench_responses_flow.py run --help` for more options.

### Evaluating responses

If hosting Gemma3-27B-it locally:

```bash
python evaluator_flow.py --environment=pypi run --responses_ds_path="local_results/responses/val_responses-gpt-4-1-nano" --inference_url="http://0.0.0.0:8000/v1"  --output_dataset_path="local_results/gpt-4-1-nano"
```

This will create
- a huggingface dataset at the path `local_results/gpt-4-1-nano-gemma-3-27b-it-evaluation` with the evaluator responses
- Data (.csv) and plots (.png) at `evaluation_results/gpt-4-1-nano-gemma-3-27b-it/*` that contain the summarized metrics and some charts.

You can use any model as an evaluator by changing setting `--judge_model_id` accordingly, including GPT, Claude, etc. (for those you don't need to set `--inference_url`). Run `python evaluator_flow.py run --help` for more options.

### Notes

- A bunch of temporary datasets that will be created in the path `local_data`.
- The API inference is designed to cache results (and currently it caches empty responses even if API requests fail). When re-running with same parameters, clear out `local_data` first.
- For `output_dataset_path`, you can use any path starting with `local` or use a HuggingFace hosted dataset repo id.
