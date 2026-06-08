import argparse
import json
import os

from datasets import load_dataset


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


def download_wildguard(train_out: str, max_train: int | None) -> None:
    print("Loading allenai/wildguardmix train split …")
    ds       = load_dataset("allenai/wildguardmix", "wildguardtrain", trust_remote_code=True)
    train_ds = ds["train"]
    if max_train is not None:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    keep_cols = ["prompt", "response", "prompt_harm_label", "response_refusal_label", "response_harm_label", "adversarial", "subcategory"]
    rows = [{col: row[col] for col in keep_cols if col in row} for row in train_ds]
    write_jsonl(rows, train_out)


def download_jailbreakbench(out_path: str) -> None:
    print("Loading JailbreakBench/JBB-Behaviors …")
    ds         = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", trust_remote_code=True)
    harmful_ds = ds["harmful"]

    rows = []
    for row in harmful_ds:
        rows.append({
            "prompt":            row["Goal"],
            "target":            row["Target"],
            "behavior":          row["Behavior"],
            "category":          row["Category"],
            "source":            row["Source"],
            "prompt_harm_label": "harmful",
        })

    write_jsonl(rows, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_out",      default="data/wildguard/train.jsonl")
    parser.add_argument("--jbb_out",        default="data/jailbreakbench/test.jsonl")
    parser.add_argument("--max_train",      type=int, default=None)
    parser.add_argument("--wildguard_only", action="store_true")
    parser.add_argument("--jbb_only",       action="store_true")
    args = parser.parse_args()

    if not args.jbb_only:
        download_wildguard(args.train_out, args.max_train)
    if not args.wildguard_only:
        download_jailbreakbench(args.jbb_out)

    print("Done.")


if __name__ == "__main__":
    main()
