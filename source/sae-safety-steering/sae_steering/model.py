from __future__ import annotations
from typing import Callable, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ObservableLanguageModel:

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name_or_path

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
        )
        self.hf_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.d_model = self.hf_model.config.hidden_size

    def _tokenize(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(self.device)

    def _layer_name_to_module(self, hook_layer: str):
        parts = hook_layer.split(".")
        module = self.hf_model
        for part in parts:
            module = getattr(module, part)
        return module

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        interventions: Optional[dict[str, Callable]] = None,
    ) -> str:
        input_ids = self._tokenize(prompt)
        hooks = []

        if interventions:
            for hook_layer, fn in interventions.items():
                if fn is None:
                    continue
                module = self._layer_name_to_module(hook_layer)

                def make_hook(intervention_fn):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            return intervention_fn(output)
                        hidden = output[0]
                        steered = intervention_fn(hidden)
                        return (steered,) + output[1:]
                    return hook

                h = module.register_forward_hook(make_hook(fn))
                hooks.append(h)

        try:
            with torch.no_grad():
                out = self.hf_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            for h in hooks:
                h.remove()

        generated = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def get_activations(self, prompt: str, hook_layer: str) -> torch.Tensor:
        input_ids = self._tokenize(prompt)
        captured = {}

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured["acts"] = output.detach()
            else:
                captured["acts"] = output[0].detach()

        module = self._layer_name_to_module(hook_layer)
        h = module.register_forward_hook(hook)

        try:
            with torch.no_grad():
                self.hf_model(input_ids, use_cache=False)
        finally:
            h.remove()

        return captured["acts"].squeeze(0)
