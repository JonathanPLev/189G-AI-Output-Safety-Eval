"""
eval_scripts/run_eval.py

Runs both the baseline (unsteered) and steered Llama 3.1 8B on WildGuardTest
and judges every output with the OpenAI judge.

Saves results to results/eval_results.jsonl with columns:
  prompt, prompt_harm_label, wildguard_response, response_harm_label,
  baseline_response, baseline_judge_label,
  steered_response,  steered_judge_label,
  adversarial, subcategory

Usage:
    export OPENAI_API_KEY=sk-...
    python eval_scripts/run_eval.py
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae_steering.model import ObservableLanguageModel
from sae_steering.sae import load_sae
from sae_steering.features import FeatureBank
from steering_scripts.steer import build_intervention
from train_scripts.judge_outputs import run_guardreasoner, parse_guardreasoner_output, run_gpt4o


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/steering_config.yaml")
    parser.add_argument("--out",         default="results/eval_results.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg["model"]["device"]

    # ── load test data ───────────────────────────────────────────────────────
    with open(cfg["data"]["test_path"]) as f:
        test_rows = [json.loads(l) for l in f if l.strip()]

    cap = args.max_samples or cfg["data"].get("max_test_samples")
    if cap:
        test_rows = test_rows[:cap]
    print(f"Evaluating on {len(test_rows):,} test samples …")

    # ── load model + SAE + features ──────────────────────────────────────────
    model = ObservableLanguageModel(
        model_name_or_path=cfg["model"]["name"],
        device=device,
        dtype=getattr(torch, cfg["model"]["dtype"]),
    )
    sae = load_sae(
        path=cfg["sae"]["local_path"],
        d_model=model.d_model,
        expansion_factor=cfg["sae"]["expansion_factor"],
        device=torch.device(device),
    )
    bank = FeatureBank.from_yaml("configs/features.yaml")
    print(f"Loaded {bank}")

    steer_cfg = cfg["steering"]
    intervention = build_intervention(
        sae=sae,
        bank=bank,
        harmful_threshold=steer_cfg["harmful_activation_threshold"],
        ablate_harmful=steer_cfg["ablate_harmful"],
        boost_refusal=steer_cfg["boost_refusal"],
        refusal_boost_strength=steer_cfg["refusal_boost_strength"],
        unharmful_boost_strength=steer_cfg["unharmful_boost_strength"],
    )

    # ── generate baseline + steered responses ────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    baseline_rows = []
    steered_rows  = []

    for i, row in enumerate(test_rows):
        prompt = row["prompt"]

        try:
            baseline_resp = model.generate(
                prompt, max_new_tokens=cfg["eval"]["max_new_tokens"], interventions=None,
            )
        except Exception as e:
            print(f"  [warn] baseline row {i}: {e}")
            baseline_resp = ""

        try:
            steered_resp = model.generate(
                prompt, max_new_tokens=cfg["eval"]["max_new_tokens"],
                interventions={cfg["sae"]["hook_layer"]: intervention},
            )
        except Exception as e:
            print(f"  [warn] steered row {i}: {e}")
            steered_resp = ""

        baseline_rows.append({**row, "model_response": baseline_resp})
        steered_rows.append( {**row, "model_response": steered_resp})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_rows)} generated")

    # ── judge all responses with GuardReasoner (+ GPT-4o fallback) ───────────
    from openai import OpenAI
    gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gpt_model  = cfg["eval"]["judge_model"]

    def judge_batch(rows_to_judge: list[dict], tag: str) -> list[dict]:
        print(f"  Running GuardReasoner on {tag} responses …")
        results = run_guardreasoner(rows_to_judge)
        failed  = [i for i, r in enumerate(results) if r is None]
        print(f"    {len(failed)} parse failures → GPT-4o fallback")
        for idx in failed:
            results[idx] = run_gpt4o(rows_to_judge[idx], gpt_client, gpt_model)
        return results

    baseline_judgments = judge_batch(baseline_rows, "baseline")
    steered_judgments  = judge_batch(steered_rows,  "steered")

    # ── write results ─────────────────────────────────────────────────────────
    with open(args.out, "w") as out_f:
        for row, b_judge, s_judge in zip(test_rows, baseline_judgments, steered_judgments):
            out_f.write(json.dumps({
                "prompt":                row.get("prompt", ""),
                "prompt_harm_label":     row.get("prompt_harm_label", ""),
                "response_harm_label":   row.get("response_harm_label", ""),
                "wildguard_response":    row.get("response", ""),
                "adversarial":           row.get("adversarial", False),
                "subcategory":           row.get("subcategory", ""),
                "baseline_response":     baseline_rows[test_rows.index(row)]["model_response"],
                "baseline_judge_label":  b_judge.get("judge_label", "unknown"),
                "baseline_judge_source": b_judge.get("judge_source", "unknown"),
                "steered_response":      steered_rows[test_rows.index(row)]["model_response"],
                "steered_judge_label":   s_judge.get("judge_label", "unknown"),
                "steered_judge_source":  s_judge.get("judge_source", "unknown"),
            }) + "\n")

    print(f"\nEval results saved → {args.out}")


if __name__ == "__main__":
    main()
