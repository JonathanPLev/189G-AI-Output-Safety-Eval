from __future__ import annotations
import torch


class SparseAutoEncoder(torch.nn.Module):

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(f)


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
) -> SparseAutoEncoder:
    sae = SparseAutoEncoder(
        d_in=d_model,
        d_hidden=d_model * expansion_factor,
        device=device,
    )
    state_dict = torch.load(path, weights_only=True, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae
