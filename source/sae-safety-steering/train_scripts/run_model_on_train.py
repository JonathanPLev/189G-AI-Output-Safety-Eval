"""
train_scripts/run_model_on_train.py

Loads Llama 3.1 8B-Instruct and runs it on the WildGuardTrain subset.
Saves each prompt → model_response alongside the original labels to
results/train_outputs.jsonl.

These outputs are used by:
  - train_scripts/judge_outputs.py  (to get judge labels)
  - steering_scripts/identify_features.py  (to get SAE activations)

Usage:
    python train_scripts/run_model_on_train.py
    python train_scripts/run_model_on_train.py --max_samples 500
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_steering.model import ObservableLanguageModel


def load_config(path: str = "configs/steering_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/steering_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out",         default="results/train_outputs.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── load data ────────────────────────────────────────────────────────────
    rows = load_jsonl(cfg["data"]["train_path"])
    if args.max_samples:
        rows = rows[:args.max_samples]
    elif cfg["data"].get("max_train_samples"):
        rows = rows[:cfg["data"]["max_train_samples"]]

    print(f"Running model on {len(rows):,} train samples …")

    # ── load model ───────────────────────────────────────────────────────────
    model = ObservableLanguageModel(
        model_name_or_path=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
    )

    # ── generate ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as out_f:
        for i, row in enumerate(rows):
            prompt = row["prompt"]
            try:
                response = model.generate(
                    prompt,
                    max_new_tokens=cfg["eval"]["max_new_tokens"],
                    interventions=None,   # no steering — baseline outputs
                )
            except Exception as e:
                import traceback
                print(f"  [warn] row {i} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                response = ""
                break  # stop after first failure to see full traceback

            out_row = {
                "prompt":                 prompt,
                "model_response":         response,
                "wildguard_response":     row.get("response", ""),
                "prompt_harm_label":      row.get("prompt_harm_label", ""),
                "response_refusal_label": row.get("response_refusal_label", ""),
                "response_harm_label":    row.get("response_harm_label", ""),
                "adversarial":            row.get("adversarial", False),
                "subcategory":            row.get("subcategory", ""),
            }
            out_f.write(json.dumps(out_row) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()