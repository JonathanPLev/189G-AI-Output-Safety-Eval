import torch
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

pipe("The key to life is")
