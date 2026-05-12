"""
preprocess_assistments.py
-----
Cleans ASSISTments 2009-10 skill_builder_data_corrected.csv
and produces train / val / test splits + Table 1 stats.

Usage:
    python preprocess_assistments.py --input skill_builder_data_corrected.csv

Output (all in ./data/assistments/):
    train.parquet   70% by time
    val.parquet     10% by time
    test.parquet    20% by time
    stats.json      counts for Table 1
"""

import argparse, json, os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

MIN_INTERACTIONS = 10   # drop students with fewer than this many responses

def load(path):
    print(f"  Loading {path} ...")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Raw shape: {df.shape}")
    return df

def clean(df):
    print("\n[1] Cleaning ...")

    # keep only the columns we actually need
    needed = ["order_id", "user_id", "skill_name", "correct",
              "problem_id", "ms_first_response"]
    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    # drop rows missing skill or correctness
    before = len(df)
    df = df.dropna(subset=["skill_name", "correct"])
    print(f"  Dropped {before - len(df):,} rows missing skill/correct")

    # correct must be 0 or 1
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])]

    # sort by interaction order (order_id is the global sequence number)
    if "order_id" in df.columns:
        df = df.sort_values("order_id").reset_index(drop=True)

    return df

def filter_students(df):
    print("\n[2] Filtering students with < %d interactions ..." % MIN_INTERACTIONS)
    counts = df.groupby("user_id").size()
    keep = counts[counts >= MIN_INTERACTIONS].index
    before = df["user_id"].nunique()
    df = df[df["user_id"].isin(keep)].reset_index(drop=True)
    after = df["user_id"].nunique()
    print(f"  Students: {before:,} → {after:,} (removed {before-after:,})")
    return df

def temporal_split(df):
    print("\n[3] Temporal train / val / test split (70 / 10 / 20) ...")
    # split on the global row order (already sorted by order_id)
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.80)
    train = df.iloc[:t1].copy()
    val   = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test

def encode_skills(train, val, test):
    print("\n[4] Encoding skill names → integer IDs ...")
    skills = sorted(train["skill_name"].unique())
    skill2id = {s: i for i, s in enumerate(skills)}
    for split in [train, val, test]:
        split["skill_id"] = split["skill_name"].map(skill2id).fillna(-1).astype(int)
    print(f"  Skills in training set: {len(skills):,}")
    return train, val, test, skill2id

def save(train, val, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv",     index=False)
    test.to_csv(f"{out_dir}/test.csv",   index=False)
    print(f"\n  Saved splits → {out_dir}/")

def print_table1(df, train, val, test):
    """Prints the exact numbers needed to fill in Table 1 of the paper."""
    stats = {
        "dataset": "ASSISTments 2009-10",
        "total_students":      int(df["user_id"].nunique()),
        "total_interactions":  int(len(df)),
        "total_skills":        int(df["skill_name"].nunique()),
        "train_interactions":  int(len(train)),
        "val_interactions":    int(len(val)),
        "test_interactions":   int(len(test)),
        "avg_interactions_per_student": round(len(df) / df["user_id"].nunique(), 1),
        "overall_accuracy":    round(df["correct"].mean(), 4),
    }

    print("\n" + "═"*52)
    print("  TABLE 1 — ASSISTments stats  (copy into paper)")
    print("═"*52)
    for k, v in stats.items():
        print(f"  {k:<38} {v}")
    print("═"*52)
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="skill_builder_data_corrected.csv")
    parser.add_argument("--outdir", default="data/assistments")
    args = parser.parse_args()

    print("\n── ASSISTments Preprocessing -----")
    df = load(args.input)
    df = clean(df)
    df = filter_students(df)
    train, val, test = temporal_split(df)
    train, val, test, skill2id = encode_skills(train, val, test)
    save(train, val, test, args.outdir)

    stats = print_table1(df, train, val, test)
    with open(f"{args.outdir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n✓ Done.

if __name__ == "__main__":
    main()
