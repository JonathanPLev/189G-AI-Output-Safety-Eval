import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_steering.features import FeatureBank

D_HIDDEN = 65536


def load_activations(path: str) -> tuple[list[str], np.ndarray]:
    labels   = []
    features = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("prompt_harm_label") or row.get("judge_label", "")
            if label not in ("harmful", "unharmful"):
                continue
            labels.append(label)

            feat_data = row.get("features") or row.get("mean_features")
            if isinstance(feat_data, dict):
                vec = np.zeros(D_HIDDEN, dtype=np.float32)
                for idx, val in feat_data.items():
                    vec[int(idx)] = float(val)
            else:
                vec = np.array(feat_data, dtype=np.float32)
            features.append(vec)

    return labels, np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="results/train_activations.jsonl")
    parser.add_argument("--out",         default="configs/features.yaml")
    parser.add_argument("--top_k",       type=int, default=50)
    args = parser.parse_args()

    print(f"Loading activations from {args.activations} …")
    labels, features = load_activations(args.activations)

    n_harmful   = sum(1 for l in labels if l == "harmful")
    n_unharmful = sum(1 for l in labels if l == "unharmful")
    print(f"  {len(labels):,} samples  (harmful={n_harmful}, unharmful={n_unharmful})")

    if n_harmful < 5 or n_unharmful < 5:
        raise ValueError("Too few samples in one class — check your data.")

    mask_harmful   = np.array([l == "harmful"   for l in labels])
    mask_unharmful = np.array([l == "unharmful" for l in labels])

    mean_harmful   = features[mask_harmful].mean(axis=0)
    mean_unharmful = features[mask_unharmful].mean(axis=0)

    scores = mean_harmful - mean_unharmful

    harmful_indices   = [int(i) for i in np.argsort(scores)[::-1][:args.top_k].tolist()]
    unharmful_indices = [int(i) for i in np.argsort(scores)[:args.top_k].tolist()]

    print(f"\nTop-{args.top_k} features identified:")
    print(f"  harmful   : {harmful_indices[:5]} …")
    print(f"  unharmful : {unharmful_indices[:5]} …")
    print(f"  Top feature differential score: {float(scores[harmful_indices[0]]):.4f}")

    bank = FeatureBank(
        harmful_indices=harmful_indices,
        unharmful_indices=unharmful_indices,
    )
    bank.to_yaml(args.out)
    print(f"\nDone. Feature bank written → {args.out}")


if __name__ == "__main__":
    main()
