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
from train_scripts.judge_outputs import run_guardreasoner, run_gpt4o


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_metrics(rows: list[dict], dataset: str) -> None:
    from collections import Counter

    baseline = Counter(r["baseline_judge_label"] for r in rows)
    steered  = Counter(r["steered_judge_label"]  for r in rows)

    harmful = [r for r in rows if r.get("prompt_harm_label") == "harmful"]
    benign  = [r for r in rows if r.get("prompt_harm_label") == "unharmful"]

    baseline_asr = sum(1 for r in harmful if r["baseline_judge_label"] == "harmful") / max(len(harmful), 1)
    steered_asr  = sum(1 for r in harmful if r["steered_judge_label"]  == "harmful") / max(len(harmful), 1)

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}  |  Total: {len(rows)}")
    print(f"Baseline distribution: {dict(baseline)}")
    print(f"Steered  distribution: {dict(steered)}")
    print(f"\nHarmful prompts : {len(harmful)}")
    print(f"ASR  — baseline: {baseline_asr:.3f}  steered: {steered_asr:.3f}  (Δ={steered_asr - baseline_asr:+.3f})")

    if benign:
        baseline_orr = sum(1 for r in benign if r["baseline_judge_label"] == "refusal") / len(benign)
        steered_orr  = sum(1 for r in benign if r["steered_judge_label"]  == "refusal") / len(benign)
        print(f"Benign prompts  : {len(benign)}")
        print(f"ORR  — baseline: {baseline_orr:.3f}  steered: {steered_orr:.3f}  (Δ={steered_orr - baseline_orr:+.3f})")
    else:
        print("ORR  — N/A (no benign prompts in this dataset)")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="configs/steering_config.yaml")
    parser.add_argument("--dataset",         default="wildguard", choices=["wildguard", "jailbreakbench"])
    parser.add_argument("--generations_out", default=None)
    parser.add_argument("--out",             default=None)
    parser.add_argument("--max_samples",     type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg["model"]["device"]
    os.makedirs("results", exist_ok=True)

    if args.dataset == "jailbreakbench":
        data_path    = "data/jailbreakbench/test.jsonl"
        gen_path     = args.generations_out or "results/jbb_generations.jsonl"
        results_path = args.out             or "results/jbb_results.jsonl"
    else:
        data_path    = cfg["data"]["test_path"]
        gen_path     = args.generations_out or "results/eval_generations.jsonl"
        results_path = args.out             or "results/eval_results.jsonl"

    if args.skip_generation:
        if not os.path.exists(gen_path):
            raise FileNotFoundError(f"--skip_generation passed but {gen_path} does not exist.")
        print(f"Phase 1 skipped — loading from {gen_path}")
    else:
        if os.path.exists(gen_path):
            print(f"Phase 1 skipped — {gen_path} already exists")
        else:
            with open(data_path) as f:
                test_rows = [json.loads(l) for l in f if l.strip()]

            cap = args.max_samples or cfg["data"].get("max_test_samples")
            if cap:
                test_rows = test_rows[:cap]
            print(f"Phase 1: Generating on {len(test_rows):,} {args.dataset} samples …")

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

            steer_cfg    = cfg["steering"]
            intervention = build_intervention(
                sae=sae,
                bank=bank,
                harmful_threshold=steer_cfg["harmful_activation_threshold"],
                ablate_harmful=steer_cfg["ablate_harmful"],
                boost_unharmful=steer_cfg.get("boost_unharmful", True),
                unharmful_boost_strength=steer_cfg.get("unharmful_boost_strength", 4.0),
                device=device,
            )

            with open(gen_path, "w") as gen_f:
                for i, row in enumerate(test_rows):
                    prompt = row["prompt"]

                    try:
                        baseline_resp = model.generate(prompt, max_new_tokens=cfg["eval"]["max_new_tokens"], interventions=None)
                    except Exception as e:
                        print(f"  [warn] baseline row {i}: {e}")
                        baseline_resp = ""

                    try:
                        steered_resp = model.generate(prompt, max_new_tokens=cfg["eval"]["max_new_tokens"], interventions={cfg["sae"]["hook_layer"]: intervention})
                    except Exception as e:
                        print(f"  [warn] steered row {i}: {e}")
                        steered_resp = ""

                    gen_f.write(json.dumps({
                        "prompt":             prompt,
                        "prompt_harm_label":  row.get("prompt_harm_label", "harmful"),
                        "behavior":           row.get("behavior", ""),
                        "category":           row.get("category", ""),
                        "wildguard_response": row.get("response", ""),
                        "baseline_response":  baseline_resp,
                        "steered_response":   steered_resp,
                    }) + "\n")
                    gen_f.flush()

                    if (i + 1) % 10 == 0:
                        print(f"  {i + 1}/{len(test_rows)} generated")

            print(f"Generations saved → {gen_path}")

    with open(gen_path) as f:
        gen_rows = [json.loads(l) for l in f if l.strip()]

    print(f"Phase 2: Judging {len(gen_rows):,} rows …")

    from openai import OpenAI
    gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    gpt_model  = cfg["eval"]["judge_model"]

    def judge_batch(rows, resp_key, tag):
        judge_inputs = [{**r, "model_response": r[resp_key]} for r in rows]
        print(f"  Running GuardReasoner on {tag} …")
        results = run_guardreasoner(judge_inputs)
        failed  = [i for i, r in enumerate(results) if r is None]
        print(f"    {len(failed)} parse failures → GPT-4o fallback")
        for idx in failed:
            results[idx] = run_gpt4o(judge_inputs[idx], gpt_client, gpt_model)
        return results

    baseline_judgments = judge_batch(gen_rows, "baseline_response", "baseline")
    steered_judgments  = judge_batch(gen_rows, "steered_response",  "steered")

    with open(results_path, "w") as out_f:
        for row, b_j, s_j in zip(gen_rows, baseline_judgments, steered_judgments):
            out_f.write(json.dumps({
                "prompt":                row["prompt"],
                "prompt_harm_label":     row.get("prompt_harm_label", "harmful"),
                "behavior":              row.get("behavior", ""),
                "category":              row.get("category", ""),
                "wildguard_response":    row.get("wildguard_response", ""),
                "baseline_response":     row["baseline_response"],
                "baseline_judge_label":  b_j.get("judge_label", "unknown"),
                "baseline_judge_source": b_j.get("judge_source", "unknown"),
                "steered_response":      row["steered_response"],
                "steered_judge_label":   s_j.get("judge_label", "unknown"),
                "steered_judge_source":  s_j.get("judge_source", "unknown"),
            }) + "\n")

    print_metrics([json.loads(l) for l in open(results_path) if l.strip()], args.dataset)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
