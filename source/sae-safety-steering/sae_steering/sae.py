"""
sae_steering/sae.py

SparseAutoEncoder definition and loader.
Taken directly from the Goodfire open-source SAE demo — do not change
the architecture or weight-loading logic or the pretrained weights won't load.
"""

from __future__ import annotations

import torch


class SparseAutoEncoder(torch.nn.Module):
    """
    Two-layer SAE:
        encoder: Linear(d_in, d_hidden) → ReLU
        decoder: Linear(d_hidden, d_in)

    d_hidden = d_model * expansion_factor
      For Llama-3.1-8B-Instruct-SAE-l19: 4096 * 16 = 65 536 features
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in     = d_in
        self.d_hidden = d_hidden
        self.device   = device
        self.dtype    = dtype

        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)

        self.to(self.device, self.dtype)

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(*, d_in) → (*, d_hidden)  via  ReLU(W_enc x + b_enc)"""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(*, d_hidden) → (*, d_in)"""
        return self.decoder_linear(f)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, feature_activations)."""
        f = self.encode(x)
        return self.decode(f), f

    # ── reconstruction error ─────────────────────────────────────────────────

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the residual x - decode(encode(x)), same shape as x."""
        recon, _ = self.forward(x)
        return x - recon.detach()


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
) -> SparseAutoEncoder:
    """
    Instantiate a SparseAutoEncoder and load pretrained weights from `path`.

    Args:
        path:             local .pth file path (downloaded by start_scripts/download_sae.py)
        d_model:          hidden size of the LLM (4096 for Llama-3.1-8B)
        expansion_factor: 16 for the l19 checkpoint
        device:           torch device

    Returns:
        Loaded SparseAutoEncoder in eval mode.
    """
    sae = SparseAutoEncoder(
        d_in=d_model,
        d_hidden=d_model * expansion_factor,
        device=device,
    )
    state_dict = torch.load(path, weights_only=True, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae
