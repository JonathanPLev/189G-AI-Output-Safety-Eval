"""
train_scripts/collect_activations.py

Passes each judged train sample through Llama 3.1 8B, extracts the
layer-19 residual-stream activations, encodes them through the SAE,
and saves the mean feature-activation vector per sample alongside
its judge label.

Output: results/train_activations.jsonl
  Each row: { "judge_label": str, "mean_features": [float * 65536] }

This file is the input to steering_scripts/identify_features.py.

Note: 65536 floats per row × 5000 rows ≈ 1.2 GB — make sure you have disk space.
For memory efficiency we store bfloat16 as regular Python floats (fp32 on disk).
If you need to reduce size, set --top_k to only save the top-k activated feature
indices + values as a sparse dict instead.

Usage:
    python train_scripts/collect_activations.py
    python train_scripts/collect_activations.py --sparse --top_k 512
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_steering.model import ObservableLanguageModel
from sae_steering.sae import load_sae


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/steering_config.yaml")
    parser.add_argument("--inp",    default="results/train_outputs_judged.jsonl")
    parser.add_argument("--out",    default="results/train_activations.jsonl")
    parser.add_argument("--sparse", action="store_true",
                        help="Store sparse {index: value} dicts instead of full vectors")
    parser.add_argument("--top_k",  type=int, default=512,
                        help="Used only with --sparse: keep top-k activations per sample")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── load judged rows ─────────────────────────────────────────────────────
    rows = []
    with open(args.inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # drop rows the judge couldn't classify
    rows = [r for r in rows if r.get("judge_label") in ("harmful", "unharmful", "refusal")]
    print(f"Processing {len(rows):,} labelled rows …")

    # ── load model + SAE ─────────────────────────────────────────────────────
    device = cfg["model"]["device"]
    model = ObservableLanguageModel(
        model_name_or_path=cfg["model"]["name"],
        device=device,
        dtype=getattr(torch, cfg["model"]["dtype"]),
    )
    sae = load_sae(
        path=cfg["sae"]["local_path"],
        d_model=model.d_model,
        expansion_factor=cfg["sae"]["expansion_factor"],
        device=torch.device(device),
    )
    hook_layer = cfg["sae"]["hook_layer"]

    # ── collect ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as out_f, torch.no_grad():
        for i, row in enumerate(rows):
            try:
                # (seq_len, d_model)
                acts = model.get_activations(row["prompt"], hook_layer)
                acts = acts.to(device=torch.device(device),
                               dtype=getattr(torch, cfg["model"]["dtype"]))

                # (seq_len, d_hidden)
                features = sae.encode(acts)

                # Mean-pool over token positions → (d_hidden,)
                mean_f = features.mean(dim=0)

                if args.sparse:
                    # Keep only top-k non-zero activations
                    topk = torch.topk(mean_f, k=min(args.top_k, mean_f.shape[0]))
                    feature_data = {
                        int(idx): float(val)
                        for idx, val in zip(topk.indices.tolist(), topk.values.tolist())
                    }
                else:
                    feature_data = mean_f.float().tolist()

                out_f.write(json.dumps({
                    "prompt":                 row["prompt"],
                    "prompt_harm_label":      row.get("prompt_harm_label", ""),
                    "response_harm_label":    row.get("response_harm_label", ""),
                    "response_refusal_label": row.get("response_refusal_label", ""),
                    "judge_label":            row["judge_label"],
                    "judge_source":           row.get("judge_source", ""),
                    "adversarial":            row.get("adversarial", False),
                    "subcategory":            row.get("subcategory", ""),
                    "mean_features":          feature_data,
                }) + "\n")

            except Exception as e:
                print(f"  [warn] row {i} failed: {e}")
                continue

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()