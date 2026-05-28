# 189G-AI-Output-Safety-Eval

Benchmarks and improves safety outcomes of text generation models using representation engineering. Uses the [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) dataset and an LLM-as-judge (GPT-4o) to evaluate whether model outputs are harmful.

## Setup

```bash
uv sync
cp .env.example .env  # then fill in your OPENAI_API_KEY
```

Models must be downloaded and saved locally before running:

```python
# in infer_danger_qwen_eval.py, uncomment and run the save block once:
model.save_pretrained("Qwen-0.6b")
tokenizer.save_pretrained("Qwen-0.6b")
```

You also need to authenticate with HuggingFace to access WildGuardMix:

```bash
huggingface-cli login
```

## Running

```bash
# default: runs wildguard_llm_judge_eval.py
./run.sh

# run a specific script
./run.sh infer_danger_qwen_eval.py
./run.sh train_safety_probe.py
```

Or activate the environment manually:

```bash
source .venv/bin/activate
python wildguard_llm_judge_eval.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `wildguard_llm_judge_eval.py` | Main evaluation pipeline — generates responses from Qwen3-0.6B and judges them with GPT-4o against WildGuardMix reference responses |
| `train_safety_probe.py` | Trains a logistic regression probe on layer-12 hidden states to classify safe vs. unsafe responses |
| `infer_danger_qwen_eval.py` | Runs Qwen3-0.6B with forward hooks to capture layer activations during generation |
| `infer_danger_llama_eval.py` | Baseline Llama-3.2-1B inference via HuggingFace pipeline |
| `generate_safety_report.py` | Utility to log token throughput and latency to `metrics/benchmark_results.jsonl` |

## Output

Results are written to `metrics/` (gitignored):
- `wildguard_eval_results.jsonl` — per-item judge verdicts with reasoning
- `benchmark_results.jsonl` — generation latency/throughput logs
