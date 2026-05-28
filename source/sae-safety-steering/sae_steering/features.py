"""
sae_steering/features.py

FeatureBank: a lightweight container for the three feature-index lists
(harmful, unharmful, refusal) that are identified offline and saved to
configs/features.yaml.

This module does NOT do the identification — that lives in
steering_scripts/identify_features.py.  FeatureBank is the runtime object
that steering code loads and queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml


@dataclass
class FeatureBank:
    harmful_indices:   list[int] = field(default_factory=list)
    unharmful_indices: list[int] = field(default_factory=list)
    refusal_indices:   list[int] = field(default_factory=list)

    # ── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "FeatureBank":
        """Load feature indices from configs/features.yaml."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            harmful_indices=data.get("harmful_indices", []),
            unharmful_indices=data.get("unharmful_indices", []),
            refusal_indices=data.get("refusal_indices", []),
        )

    def to_yaml(self, path: str) -> None:
        """Save feature indices to configs/features.yaml."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {
                    "harmful_indices":   self.harmful_indices,
                    "unharmful_indices": self.unharmful_indices,
                    "refusal_indices":   self.refusal_indices,
                },
                f,
            )
        print(f"Feature bank saved → {path}")

    # ── runtime helpers ───────────────────────────────────────────────────────

    def max_harmful_activation(self, feature_acts: torch.Tensor) -> float:
        """
        Given a (seq_len, d_hidden) feature activation tensor,
        return the max activation across all harmful feature indices
        and all token positions.

        Returns 0.0 if no harmful indices are set.
        """
        if not self.harmful_indices:
            return 0.0
        idx = torch.tensor(self.harmful_indices, device=feature_acts.device)
        return feature_acts[:, idx].max().item()

    def __repr__(self) -> str:
        return (
            f"FeatureBank("
            f"harmful={len(self.harmful_indices)}, "
            f"unharmful={len(self.unharmful_indices)}, "
            f"refusal={len(self.refusal_indices)})"
        )
