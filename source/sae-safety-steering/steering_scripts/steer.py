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
    boost_unharmful: bool = True,
    unharmful_boost_strength: float = 4.0,
    device: str = "cuda",
) -> Callable:

    harmful_idx   = torch.tensor(bank.harmful_indices,   dtype=torch.long, device=device)
    unharmful_idx = torch.tensor(bank.unharmful_indices, dtype=torch.long, device=device)

    def intervention(activations: torch.Tensor) -> torch.Tensor:
        squeezed = activations.dim() == 2
        if squeezed:
            activations = activations.unsqueeze(0)

        if activations.shape[1] > 1:
            if squeezed:
                return activations.squeeze(0)
            return activations

        dtype = activations.dtype
        features = sae.encode(activations)

        if harmful_idx.numel() > 0:
            max_harmful = features[..., harmful_idx].max().item()
        else:
            max_harmful = 0.0

        if max_harmful > harmful_threshold:
            reconstructed = sae.decode(features)
            error = (activations - reconstructed).detach()
            features = features.clone()

            if ablate_harmful and harmful_idx.numel() > 0:
                features[..., harmful_idx] = 0.0

            if boost_unharmful and unharmful_idx.numel() > 0:
                features[..., unharmful_idx] = (
                    features[..., unharmful_idx] + unharmful_boost_strength
                )

            steered = sae.decode(features) + error
        else:
            reconstructed = sae.decode(features)
            error = (activations - reconstructed).detach()
            steered = reconstructed + error

        steered = steered.to(dtype=dtype)
        if squeezed:
            steered = steered.squeeze(0)
        return steered

    return intervention
