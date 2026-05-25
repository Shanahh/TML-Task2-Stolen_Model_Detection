"""
Input:
  submission_features.csv or submission_rescored_features_rescored.csv

Outputs:
  score_distribution.png
  score_distribution_topk.png
  score_components_topk.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def plot_score_distribution(df, out_dir: Path):
    scores = df["score"]

    plt.figure(figsize=(7.0, 4.2))
    plt.hist(scores, bins=30, edgecolor="black", alpha=0.8)
    plt.xlabel("Final stealing-confidence score")
    plt.ylabel("Number of suspect models")
    plt.title("Distribution of final signal scores")
    plt.tight_layout()

    out_path = out_dir / "score_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_ranked_scores(df, out_dir: Path, top_k: int):
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(range(1, len(ranked) + 1), ranked["score"], marker=".", linewidth=1)
    plt.xlabel("Ranked suspect models")
    plt.ylabel("Final stealing-confidence score")
    plt.title("Ranked final scores")
    plt.tight_layout()

    out_path = out_dir / "score_rank_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")

    top = ranked.head(top_k)

    plt.figure(figsize=(8.0, 4.5))
    plt.bar(top["id"].astype(str), top["score"])
    plt.xlabel("Suspect model id")
    plt.ylabel("Final stealing-confidence score")
    plt.title(f"Top {top_k} suspect models by score")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()

    out_path = out_dir / "score_distribution_topk.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_component_scores(df, out_dir: Path, top_k: int):
    component_cols = [
        "dbg_direct_score",
        "dbg_finetune_score",
        "dbg_distilled_score",
        "dbg_boundary_score",
        "dbg_dataset_score",
    ]

    available = [c for c in component_cols if c in df.columns]
    if not available:
        print("Skipping component plot: no debug specialist score columns found.")
        return

    ranked = df.sort_values("score", ascending=False).head(top_k).copy()
    ranked["id"] = ranked["id"].astype(str)

    x = range(len(ranked))
    bottom = [0.0] * len(ranked)

    plt.figure(figsize=(9.0, 4.8))
    for col in available:
        values = ranked[col].values
        plt.bar(x, values, bottom=bottom, label=col.replace("dbg_", "").replace("_score", ""))
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xticks(x, ranked["id"], rotation=60, ha="right")
    plt.xlabel("Suspect model id")
    plt.ylabel("Specialist score sum")
    plt.title(f"Specialist score components for top {top_k} suspects")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / "score_components_topk.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="submission_features.csv",
        help="Path to submission_features.csv or diagnostics CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked models to show in top-k plots.",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path)
    require_columns(df, ["id", "score"])

    print(f"Loaded {len(df)} rows from {features_path}")
    print(f"Score range: {df['score'].min():.4f} to {df['score'].max():.4f}")

    plot_score_distribution(df, out_dir)
    plot_ranked_scores(df, out_dir, args.top_k)
    plot_component_scores(df, out_dir, args.top_k)


if __name__ == "__main__":
    main()