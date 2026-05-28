"""
steering_scripts/identify_features.py

Reads results/train_activations.jsonl (produced by
train_scripts/collect_activations.py) and identifies the SAE feature indices
that are most discriminative for each class:
  - harmful
  - unharmful
  - refusal

Method: for each class C, compute
    score(f) = mean_activation(f | label=C) - mean_activation(f | label≠C)
Then take the top-k features by score.

Writes the identified indices to configs/features.yaml so that
steering_scripts/tune_steering.py and eval_scripts/ can load them.

Usage:
    python steering_scripts/identify_features.py
    python steering_scripts/identify_features.py --top_k 30
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_steering.features import FeatureBank


VALID_LABELS = ("harmful", "unharmful", "refusal")


def load_activations(path: str) -> tuple[list[str], np.ndarray | list[dict]]:
    """
    Returns (labels, feature_matrix).

    If rows contain dict mean_features (sparse mode), returns list of dicts.
    If rows contain list mean_features (dense mode), returns np.ndarray.
    """
    labels = []
    features = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["judge_label"] not in VALID_LABELS:
                continue
            labels.append(row["judge_label"])
            features.append(row["mean_features"])

    is_sparse = isinstance(features[0], dict)
    if not is_sparse:
        return labels, np.array(features, dtype=np.float32)
    return labels, features   # list of sparse dicts


def dense_from_sparse(sparse_list: list[dict], d_hidden: int) -> np.ndarray:
    mat = np.zeros((len(sparse_list), d_hidden), dtype=np.float32)
    for i, d in enumerate(sparse_list):
        for idx, val in d.items():
            mat[i, int(idx)] = float(val)
    return mat


def top_k_differential(
    feature_matrix: np.ndarray,
    labels: list[str],
    target_label: str,
    top_k: int,
) -> list[int]:
    """
    Return the top-k feature indices most activated for `target_label`
    relative to all other labels.
    """
    mask_pos = np.array([l == target_label for l in labels])
    mask_neg = ~mask_pos

    mean_pos = feature_matrix[mask_pos].mean(axis=0)
    mean_neg = feature_matrix[mask_neg].mean(axis=0)

    scores = mean_pos - mean_neg
    top_indices = np.argsort(scores)[::-1][:top_k].tolist()
    return [int(i) for i in top_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="results/train_activations.jsonl")
    parser.add_argument("--out",         default="configs/features.yaml")
    parser.add_argument("--top_k",       type=int, default=20)
    parser.add_argument("--d_hidden",    type=int, default=65536,
                        help="SAE hidden dim (4096 * 16 for l19)")
    args = parser.parse_args()

    print(f"Loading activations from {args.activations} …")
    labels, features = load_activations(args.activations)

    # convert sparse → dense if needed
    if isinstance(features, list):
        print("Detected sparse activations — converting to dense …")
        features = dense_from_sparse(features, args.d_hidden)

    print(f"  {len(labels):,} samples, feature dim = {features.shape[1]:,}")
    for lbl in VALID_LABELS:
        n = sum(1 for l in labels if l == lbl)
        print(f"  {lbl:12s}: {n:,} samples")

    # ── differential feature ranking ─────────────────────────────────────────
    bank = FeatureBank(
        harmful_indices=top_k_differential(features, labels, "harmful",   args.top_k),
        unharmful_indices=top_k_differential(features, labels, "unharmful", args.top_k),
        refusal_indices=top_k_differential(features, labels, "refusal",   args.top_k),
    )

    print(f"\nTop-{args.top_k} features identified:")
    print(f"  harmful   : {bank.harmful_indices[:5]} …")
    print(f"  unharmful : {bank.unharmful_indices[:5]} …")
    print(f"  refusal   : {bank.refusal_indices[:5]} …")

    bank.to_yaml(args.out)
    print(f"\nDone. Feature bank written → {args.out}")


if __name__ == "__main__":
    main()
