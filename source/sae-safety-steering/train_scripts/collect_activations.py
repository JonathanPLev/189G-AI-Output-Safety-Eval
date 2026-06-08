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
    parser.add_argument("--inp",    default="data/adversarial/train.jsonl")
    parser.add_argument("--out",    default="results/train_activations.jsonl")
    parser.add_argument("--top_k",  type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    rows = []
    with open(args.inp) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prompt_harm_label") in ("harmful", "unharmful"):
                rows.append(row)

    if args.max_samples:
        rows = rows[:args.max_samples]
    elif cfg["data"].get("max_train_samples"):
        rows = rows[:cfg["data"]["max_train_samples"]]

    n_harmful   = sum(1 for r in rows if r["prompt_harm_label"] == "harmful")
    n_unharmful = sum(1 for r in rows if r["prompt_harm_label"] == "unharmful")
    print(f"Processing {len(rows):,} rows  (harmful={n_harmful}, unharmful={n_unharmful}) …")

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as out_f, torch.no_grad():
        for i, row in enumerate(rows):
            try:
                acts = model.get_activations(row["prompt"], hook_layer)
                acts = acts.to(
                    device=torch.device(device),
                    dtype=getattr(torch, cfg["model"]["dtype"])
                )

                last_token_acts = acts[-1, :]
                features = sae.encode(last_token_acts.unsqueeze(0)).squeeze(0)

                topk = torch.topk(features, k=min(args.top_k, features.shape[0]))
                feature_data = {
                    int(idx): float(val)
                    for idx, val in zip(topk.indices.tolist(), topk.values.tolist())
                    if float(val) > 0.0
                }

                out_f.write(json.dumps({
                    "prompt":            row["prompt"],
                    "prompt_harm_label": row["prompt_harm_label"],
                    "adversarial":       row.get("adversarial", False),
                    "subcategory":       row.get("subcategory", ""),
                    "features":          feature_data,
                }) + "\n")

            except Exception as e:
                print(f"  [warn] row {i} failed: {e}")
                continue

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
