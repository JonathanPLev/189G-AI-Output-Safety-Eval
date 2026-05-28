"""
start_scripts/download_data.py

Downloads allenai/wildguardmix (train + test) from HuggingFace and writes
each split to data/wildguard/{train,test}.jsonl.

Only keeps columns used downstream:
  prompt, response, prompt_harm_label, response_refusal_label,
  response_harm_label, adversarial, subcategory

Usage:
    python start_scripts/download_data.py
    python start_scripts/download_data.py --max_train 5000 --max_test 1000
"""

import argparse
import json
import os
import sys

from datasets import load_dataset


KEEP_COLS = [
    "prompt",
    "response",
    "prompt_harm_label",
    "response_refusal_label",
    "response_harm_label",
    "adversarial",
    "subcategory",
]


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_out", default="data/wildguard/train.jsonl")
    parser.add_argument("--test_out", default="data/wildguard/test.jsonl")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Cap number of train rows (None = all ~86k)")
    parser.add_argument("--max_test", type=int, default=None,
                        help="Cap number of test rows (None = all)")
    args = parser.parse_args()

    print("Loading allenai/wildguardmix from HuggingFace …")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", trust_remote_code=True)

    # ── train split ──────────────────────────────────────────────────────────
    train_ds = ds["train"]
    if args.max_train is not None:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))

    train_rows = []
    for row in train_ds:
        train_rows.append({col: row[col] for col in KEEP_COLS if col in row})

    write_jsonl(train_rows, args.train_out)

    # ── test split ───────────────────────────────────────────────────────────
    # wildguardmix exposes a test split via the wildguardtest config
    ds_test = load_dataset("allenai/wildguardmix", "wildguardtest", trust_remote_code=True)
    test_ds = ds_test["test"]
    if args.max_test is not None:
        test_ds = test_ds.select(range(min(args.max_test, len(test_ds))))

    test_rows = []
    for row in test_ds:
        test_rows.append({col: row[col] for col in KEEP_COLS if col in row})

    write_jsonl(test_rows, args.test_out)

    print("Done.")


if __name__ == "__main__":
    main()
