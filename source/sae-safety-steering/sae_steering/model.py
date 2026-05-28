"""
sae_steering/model.py

Thin wrapper around nnsight.LanguageModel that supports:
  - caching intermediate activations at named hook points
  - injecting arbitrary interventions at named hook points
  - autoregressive token generation with optional steering

Faithful to the Goodfire open-source SAE demo pattern:
  - save full tensors inside trace context
  - index/slice AFTER the trace exits
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import nnsight


class ObservableLanguageModel:

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self._model_id = model_name_or_path

        self._model = nnsight.LanguageModel(
            model_name_or_path,
            device_map=device,
            torch_dtype=dtype,
        )

        # nnsight is lazy — run a tiny trace to force weight download/load.
        # Must pass a batched tensor (1, seq_len), not a flat list of ints.
        _warmup = self._model.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            return_tensors="pt",
        )
        if hasattr(_warmup, "input_ids"):
            _warmup = _warmup.input_ids
        _warmup = _warmup.to(device)
        with self._model.trace(_warmup):
            pass

        self.tokenizer = self._model.tokenizer
        self.d_model = self._infer_d_model()
        self.safe_mode = False

    def _infer_d_model(self) -> int:
        cfg = self._model.config
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise RuntimeError("Cannot infer hidden_size from model config.")

    def _find_module(self, hook_point: str):
        parts = hook_point.split(".")
        module = self._model
        for part in parts:
            module = getattr(module, part)
        return module

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_activations_at: Optional[list[str]] = None,
        interventions: Optional[dict[str, Callable]] = None,
    ) -> tuple[torch.Tensor, object, dict[str, torch.Tensor]]:
        """
        Single forward pass.

        Returns:
            logits   : (vocab_size,) — last token position only
            kv_cache : past_key_values
            cache    : dict[hook_point → (seq_len, d_model)]
        """
        activation_cache: dict[str, torch.Tensor] = {}

        with self._model.trace(
            input_ids,
            scan=self.safe_mode,
            validate=self.safe_mode,
        ):
            if interventions:
                for hook_point, fn in interventions.items():
                    if fn is None:
                        continue
                    module = self._find_module(hook_point)
                    intervened = fn(module.output[0])
                    module.output = (intervened,)

            if cache_activations_at:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    activation_cache[hook_point] = module.output.save()

            # Save full logits tensor — index AFTER trace exits (nnsight proxy limitation)
            all_logits = self._model.output[0].save()
            kv_cache   = self._model.output.past_key_values.save()

        # all_logits is (batch, seq_len, vocab_size) — take last token, first batch
        last_logits = all_logits.detach()[0, -1, :]   # (vocab_size,)

        return (
            last_logits,
            kv_cache,
            {k: v[0].detach() for k, v in activation_cache.items()},
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        interventions: Optional[dict[str, Callable]] = None,
    ) -> str:
        """Greedy autoregressive generation with optional steering."""
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template may return BatchEncoding — extract the tensor
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        input_ids = input_ids.to(self.device)

        prompt_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(input_ids, interventions=interventions)
            # logits is (vocab_size,) — argmax gives a scalar tensor
            new_token = logits.argmax(-1)

            if new_token.item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat(
                [input_ids[0], new_token.unsqueeze(0)], dim=0
            ).unsqueeze(0)

        generated_ids = input_ids[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_activations(self, prompt: str, hook_layer: str) -> torch.Tensor:
        """Return (seq_len, d_model) residual-stream tensor at hook_layer."""
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        input_ids = input_ids.to(self.device)

        _, _, cache = self.forward(input_ids, cache_activations_at=[hook_layer])
        return cache[hook_layer]  # (seq_len, d_model)