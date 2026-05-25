#!/usr/bin/env python3
"""
Visualize which signal family helped most in the stolen-model detector.

Input:
  submission_diagnostics.csv

Outputs:
  signal_family_distribution.png
  signal_family_topk_distribution.png
  signal_family_score_boxplot.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SPECIALIST_COLUMNS = [
    "dbg_direct_score",
    "dbg_finetune_score",
    "dbg_distilled_score",
    "dbg_boundary_score",
    "dbg_dataset_score",
]


def add_best_specialist_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    available = [c for c in SPECIALIST_COLUMNS if c in df.columns]
    if not available:
        raise ValueError(
            "No specialist score columns found. Expected at least one of: "
            + ", ".join(SPECIALIST_COLUMNS)
        )

    if "dbg_best_specialist" not in df.columns:
        df["dbg_best_specialist"] = (
            df[available]
            .idxmax(axis=1)
            .str.replace("dbg_", "", regex=False)
            .str.replace("_score", "", regex=False)
        )

    return df


def plot_best_specialist_distribution(df: pd.DataFrame, out_dir: Path):
    counts = df["dbg_best_specialist"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Dominant signal family")
    plt.ylabel("Number of suspect models")
    plt.title("Distribution of dominant stolen-model detection signals")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = out_dir / "signal_family_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_topk_best_specialist_distribution(df: pd.DataFrame, out_dir: Path, top_k: int):
    top = df.sort_values("score", ascending=False).head(top_k)
    counts = top["dbg_best_specialist"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Dominant signal family")
    plt.ylabel(f"Number of models in top {top_k}")
    plt.title(f"Dominant signals among top {top_k} scored suspects")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = out_dir / "signal_family_topk_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_specialist_score_distributions(df: pd.DataFrame, out_dir: Path):
    available = [c for c in SPECIALIST_COLUMNS if c in df.columns]

    labels = [
        c.replace("dbg_", "").replace("_score", "")
        for c in available
    ]

    values = [df[c].dropna().values for c in available]

    plt.figure(figsize=(7.5, 4.2))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.xlabel("Signal family")
    plt.ylabel("Specialist score")
    plt.title("Distribution of specialist signal scores")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = out_dir / "signal_family_score_boxplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_mean_specialist_scores_top_vs_rest(df: pd.DataFrame, out_dir: Path, top_k: int):
    available = [c for c in SPECIALIST_COLUMNS if c in df.columns]

    ranked = df.sort_values("score", ascending=False)
    top = ranked.head(top_k)
    rest = ranked.iloc[top_k:]

    top_means = top[available].mean()
    rest_means = rest[available].mean()

    labels = [
        c.replace("dbg_", "").replace("_score", "")
        for c in available
    ]

    x = range(len(available))
    width = 0.35

    plt.figure(figsize=(7.5, 4.2))
    plt.bar([i - width / 2 for i in x], top_means.values, width=width, label=f"Top {top_k}")
    plt.bar([i + width / 2 for i in x], rest_means.values, width=width, label="Rest")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.xlabel("Signal family")
    plt.ylabel("Mean specialist score")
    plt.title(f"Signal strength: top {top_k} suspects vs rest")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "signal_family_top_vs_rest.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="submission_diagnostics.csv",
        help="Path to submission_diagnostics.csv or a feature CSV containing debug specialist columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures",
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-ranked models used for top-k signal analysis.",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path)

    if "score" not in df.columns:
        raise ValueError("Input CSV must contain a 'score' column.")

    df = add_best_specialist_if_missing(df)

    print(f"Loaded {len(df)} rows from {features_path}")
    print("\nDominant signal family counts:")
    print(df["dbg_best_specialist"].value_counts().to_string())

    print(f"\nDominant signal family counts in top {args.top_k}:")
    print(
        df.sort_values("score", ascending=False)
        .head(args.top_k)["dbg_best_specialist"]
        .value_counts()
        .to_string()
    )

    plot_best_specialist_distribution(df, out_dir)
    plot_topk_best_specialist_distribution(df, out_dir, args.top_k)
    plot_specialist_score_distributions(df, out_dir)
    plot_mean_specialist_scores_top_vs_rest(df, out_dir, args.top_k)


if __name__ == "__main__":
    main()