"""
steering_scripts/steer.py

Builds the intervention callable that is passed into
ObservableLanguageModel.generate(interventions={...}).

The intervention:
  1. Encodes activations through the SAE
  2. Computes the reconstruction error (must be added back — see Goodfire demo)
  3. Checks if any harmful features are strongly active
  4. If so:
       a. Ablates (zeros) harmful features
       b. Boosts refusal and/or unharmful features
  5. Decodes back and adds the reconstruction error term

This is applied at every autoregressive step so steering persists throughout
the generated sequence.
"""

from __future__ import annotations

from typing import Callable

import torch

from sae_steering.sae import SparseAutoEncoder
from sae_steering.features import FeatureBank


def build_intervention(
    sae: SparseAutoEncoder,
    bank: FeatureBank,
    harmful_threshold: float = 1.0,
    ablate_harmful: bool = True,
    boost_refusal: bool = True,
    refusal_boost_strength: float = 8.0,
    unharmful_boost_strength: float = 4.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns an intervention function compatible with ObservableLanguageModel.forward().

    The returned function takes a (batch, seq_len, d_model) activation tensor
    and returns a steered tensor of the same shape.

    Args:
        sae:                      loaded SparseAutoEncoder
        bank:                     FeatureBank with identified feature indices
        harmful_threshold:        if max harmful feature activation > this, intervene
        ablate_harmful:           zero out harmful feature activations
        boost_refusal:            amplify refusal feature activations
        refusal_boost_strength:   additive boost applied to refusal features
        unharmful_boost_strength: additive boost applied to unharmful features
    """

    harmful_idx   = torch.tensor(bank.harmful_indices,   dtype=torch.long)
    unharmful_idx = torch.tensor(bank.unharmful_indices, dtype=torch.long)
    refusal_idx   = torch.tensor(bank.refusal_indices,   dtype=torch.long)

    def intervention(activations: torch.Tensor) -> torch.Tensor:
        """
        activations: (batch, seq_len, d_model)  — the residual stream at layer 19
        returns:     (batch, seq_len, d_model)  — steered residual stream
        """
        device = activations.device
        dtype  = activations.dtype

        # Move index tensors to the right device
        h_idx = harmful_idx.to(device)
        u_idx = unharmful_idx.to(device)
        r_idx = refusal_idx.to(device)

        # ── encode ────────────────────────────────────────────────────────────
        # features: (batch, seq_len, d_hidden)
        features = sae.encode(activations)

        # ── compute reconstruction error (must add back after decoding) ───────
        # Critical — see Goodfire demo note: "Very important to add the error term back in!"
        reconstructed = sae.decode(features)
        error = activations - reconstructed.detach()

        # ── check if intervention is needed ───────────────────────────────────
        # Use max harmful activation across batch × seq_len × harmful_features
        if h_idx.numel() > 0:
            max_harmful = features[..., h_idx].max().item()
        else:
            max_harmful = 0.0

        if max_harmful > harmful_threshold:
            # ── ablate harmful features ───────────────────────────────────────
            if ablate_harmful and h_idx.numel() > 0:
                features = features.clone()
                features[..., h_idx] = 0.0

            # ── boost refusal features ────────────────────────────────────────
            if boost_refusal and r_idx.numel() > 0:
                features = features.clone() if ablate_harmful else features
                features[..., r_idx] = features[..., r_idx] + refusal_boost_strength

            # ── boost unharmful features ──────────────────────────────────────
            if u_idx.numel() > 0:
                features[..., u_idx] = features[..., u_idx] + unharmful_boost_strength

        # ── decode and add error back ─────────────────────────────────────────
        steered = sae.decode(features) + error
        return steered.to(dtype=dtype)

    return intervention
