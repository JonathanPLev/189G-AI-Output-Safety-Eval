# SAE Safety Steering

Inference-time safety steering for Llama 3.1 8B using Goodfire's open-source
Sparse Autoencoder (SAE) trained on layer 19 of the model.

The core idea: encode residual-stream activations through the SAE, identify
which latent features correspond to harmful / unharmful / refusal behaviours,
then at inference time ablate harmful features and boost refusal/unharmful
features when harmful activation is detected.

---

## Setup

```bash
pip install -e ".[notebooks]"
```

You'll need:
- A HuggingFace token with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- An OpenAI API key (for the judge)
- A Goodfire API key (optional — only needed if you want to look up feature labels)

```bash
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...
export GOODFIRE_API_KEY=...   # optional
```

---

## Pipeline

Run these steps in order:

### 1. Download data and SAE weights

```bash
python start_scripts/download_data.py --max_train 5000 --max_test 1000
python start_scripts/download_sae.py
```

### 2. Run Llama 3.1 8B on WildGuardTrain (baseline, no steering)

```bash
python train_scripts/run_model_on_train.py
```

Output: `results/train_outputs.jsonl`

### 3. Judge the train outputs

```bash
python train_scripts/judge_outputs.py
```

Output: `results/train_outputs_judged.jsonl`
Each row gets a `judge_label` in {harmful, unharmful, refusal}.

### 4. Collect SAE feature activations for each labelled sample

```bash
# Dense mode (large but simple):
python train_scripts/collect_activations.py

# Sparse mode (saves disk space, keeps top-512 features per sample):
python train_scripts/collect_activations.py --sparse --top_k 512
```

Output: `results/train_activations.jsonl`

### 5. Identify discriminative features per class

```bash
python steering_scripts/identify_features.py --top_k 20
```

Output: `configs/features.yaml` — the harmful / unharmful / refusal feature indices.

### 6. Tune steering hyperparameters on a val split

```bash
python steering_scripts/tune_steering.py --val_size 200
```

Grid-searches threshold + boost strengths. Best params written back to
`configs/steering_config.yaml`.

### 7. Evaluate: steered vs. unsteered on WildGuardTest

```bash
python eval_scripts/run_eval.py
```

Output: `results/eval_results.jsonl`

### 8. Compute metrics

```bash
python eval_scripts/metrics.py
```

Prints ASR, ORR, F1 for baseline and steered, plus Δ metrics.
Output: `results/metrics_summary.json`

---

## Project structure

```
sae-safety-steering/
├── configs/
│   ├── steering_config.yaml   # model paths, steering params, eval settings
│   └── features.yaml          # identified feature indices (auto-generated)
├── data/
│   └── wildguard/
│       ├── train.jsonl
│       └── test.jsonl
├── models/
│   ├── llama/                 # HF model cache
│   └── sae/                   # Goodfire SAE .pth weights
├── results/                   # all outputs written here
├── start_scripts/
│   ├── download_data.py
│   └── download_sae.py
├── sae_steering/              # core package
│   ├── model.py               # ObservableLanguageModel (nnsight wrapper)
│   ├── sae.py                 # SparseAutoEncoder + load_sae
│   └── features.py            # FeatureBank container
├── train_scripts/
│   ├── run_model_on_train.py  # generate baseline outputs on train set
│   ├── judge_outputs.py       # OpenAI judge classifies outputs
│   └── collect_activations.py # extract + encode SAE features per sample
├── steering_scripts/
│   ├── identify_features.py   # differential mean to find feature indices
│   ├── steer.py               # builds the nnsight intervention callable
│   └── tune_steering.py       # grid search on val set
└── eval_scripts/
    ├── run_eval.py             # steered vs baseline on test set
    └── metrics.py             # ASR / ORR / F1 summary
```

---

## Key metrics

| Metric | Description | Target |
|--------|-------------|--------|
| ASR    | Attack Success Rate — fraction of harmful prompts that still get harmful responses | ↓ lower |
| ORR    | Over-Refusal Rate — fraction of benign prompts that get refused | ↓ lower |
| F1     | Harmless-class F1 on safety labels | ↑ higher |

---

## References

- [Goodfire SAE open-source announcement](https://www.goodfire.ai/blog/sae-open-source-announcement/)
- [WildGuardMix dataset](https://huggingface.co/datasets/allenai/wildguardmix)
- [nnsight](https://nnsight.net)
