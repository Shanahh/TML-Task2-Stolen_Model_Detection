#!/usr/bin/env python3
"""
Visualize feature-family contributions for the stolen-model detector.

Input:
  submission_diagnostics.csv

Outputs:
  feature_type_distribution_all.png
  feature_type_distribution_topk.png
  feature_type_score_boxplot.png
  feature_type_top_vs_rest.png
  feature_type_heatmap_topk.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "Weight cosine": [
        "dbg_weight",
        "w_cos_all",
        "w_cos_conv",
        "w_cos_bn",
        "w_cos_fc",
        "w_cos_early",
        "w_cos_mid",
        "w_cos_late",
        "w_sign",
    ],
    "CKA representation": [
        "dbg_cka",
        "cka_mean",
        "cka_early",
        "cka_mid",
        "cka_late",
        "cka_layer1",
        "cka_layer2",
        "cka_layer3",
        "cka_layer4",
        "cka_avgpool",
    ],
    "Clean functional": [
        "dbg_clean_dark",
        "test_logit_cos",
        "test_top1_agree",
        "test_neg_js",
        "test_conf_corr",
        "test_top5_overlap",
        "main_neg_js",
        "nonmain_neg_js",
    ],
    "Mistake agreement": [
        "dbg_mistake",
        "test_same_wrong",
        "main_same_wrong",
        "nonmain_same_wrong",
        "confwrong_top1_agree",
        "vconfwrong_top1_agree",
        "confwrong_logit_cos",
        "confwrong_neg_js",
    ],
    "High-leakage probes": [
        "dbg_leakage",
        "lowmargin_logit_cos",
        "lowmargin_neg_js",
        "lowmargin_top5_overlap",
        "highentropy_logit_cos",
        "highentropy_neg_js",
        "leakage_logit_cos",
        "leakage_neg_js",
    ],
    "OOD probes": [
        "dbg_ood",
        "ood_logit_cos",
        "ood_neg_js",
        "ood_top5_overlap",
        "ood_conf_corr",
    ],
    "Transform sensitivity": [
        "dbg_transform",
        "transform_delta_cos",
        "transform_stability_agree",
        "mixmatch_cos",
        "mixmatch_neg_js",
        "mixmatch_top1",
    ],
    "Train-subset alignment": [
        "dbg_mem",
        "mem_loss_gap_similarity",
        "mem_conf_gap_similarity",
        "mem_acc_gap_similarity",
        "mem_agree_gap_similarity",
    ],
    "FGSM boundary": [
        "dbg_fgsm",
        "fgsm_logit_cos",
        "fgsm_top1_agree",
        "fgsm_neg_js",
    ],
    "Jacobian gradient": [
        "dbg_jac",
        "jacobian_cos",
        "jacobian_sign",
    ],
}


def rank01(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)

    if not finite.any():
        return np.zeros_like(values)

    med = np.nanmedian(values[finite])
    values = np.nan_to_num(
        values,
        nan=med,
        posinf=np.nanmax(values[finite]),
        neginf=np.nanmin(values[finite]),
    )

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    if len(values) > 1:
        ranks = ranks / (len(values) - 1)

    return ranks


def build_feature_group_scores(df):
    out = pd.DataFrame()
    out["id"] = df["id"]
    out["score"] = df["score"]

    for group_name, cols in FEATURE_GROUPS.items():
        available = [c for c in cols if c in df.columns]

        if not available:
            continue

        # Prefer debug aggregate columns if they exist.
        debug_cols = [c for c in available if c.startswith("dbg_")]
        if debug_cols:
            out[group_name] = df[debug_cols].mean(axis=1)
        else:
            ranked_cols = [rank01(df[c].values) for c in available]
            out[group_name] = np.mean(np.vstack(ranked_cols), axis=0)

    feature_cols = [c for c in out.columns if c not in ["id", "score"]]

    if not feature_cols:
        raise ValueError("No usable feature-group columns found.")

    out["best_feature_type"] = out[feature_cols].idxmax(axis=1)
    out["best_feature_type_score"] = out[feature_cols].max(axis=1)

    return out, feature_cols


def plot_feature_type_distribution(group_df, out_dir, top_k=None):
    data = group_df

    if top_k is not None:
        data = group_df.sort_values("score", ascending=False).head(top_k)
        title = f"Dominant feature types among top {top_k} suspects"
        filename = "feature_type_distribution_topk.png"
        ylabel = f"Number of models in top {top_k}"
    else:
        title = "Dominant feature types across all suspects"
        filename = "feature_type_distribution_all.png"
        ylabel = "Number of suspect models"

    counts = data["best_feature_type"].value_counts()

    plt.figure(figsize=(9, 4.5))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Dominant feature type")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_feature_type_boxplot(group_df, feature_cols, out_dir):
    values = [group_df[c].dropna().values for c in feature_cols]

    plt.figure(figsize=(9, 4.5))
    plt.boxplot(values, labels=feature_cols, showfliers=False)
    plt.xlabel("Feature type")
    plt.ylabel("Rank-normalized signal strength")
    plt.title("Distribution of feature-type signal strengths")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out_path = out_dir / "feature_type_score_boxplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_top_vs_rest(group_df, feature_cols, out_dir, top_k):
    ranked = group_df.sort_values("score", ascending=False)
    top = ranked.head(top_k)
    rest = ranked.iloc[top_k:]

    top_means = top[feature_cols].mean()
    rest_means = rest[feature_cols].mean()

    x = np.arange(len(feature_cols))
    width = 0.35

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, top_means.values, width=width, label=f"Top {top_k}")
    plt.bar(x + width / 2, rest_means.values, width=width, label="Rest")
    plt.xticks(x, feature_cols, rotation=35, ha="right")
    plt.xlabel("Feature type")
    plt.ylabel("Mean rank-normalized signal")
    plt.title(f"Feature-type strength: top {top_k} suspects vs rest")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "feature_type_top_vs_rest.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def plot_topk_heatmap(group_df, feature_cols, out_dir, top_k):
    ranked = group_df.sort_values("score", ascending=False).head(top_k)

    matrix = ranked[feature_cols].values
    labels = ranked["id"].astype(str).tolist()

    plt.figure(figsize=(10, max(4, top_k * 0.28)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Rank-normalized signal strength")
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=35, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Feature type")
    plt.ylabel("Suspect model id")
    plt.title(f"Feature-type signals for top {top_k} suspects")
    plt.tight_layout()

    out_path = out_dir / "feature_type_heatmap_topk.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def write_summary_csv(group_df, feature_cols, out_dir, top_k):
    ranked = group_df.sort_values("score", ascending=False)

    summary = pd.DataFrame({
        "feature_type": feature_cols,
        "mean_all": [group_df[c].mean() for c in feature_cols],
        f"mean_top_{top_k}": [ranked.head(top_k)[c].mean() for c in feature_cols],
        "mean_rest": [ranked.iloc[top_k:][c].mean() for c in feature_cols],
        "best_type_count_all": [
            (group_df["best_feature_type"] == c).sum() for c in feature_cols
        ],
        f"best_type_count_top_{top_k}": [
            (ranked.head(top_k)["best_feature_type"] == c).sum() for c in feature_cols
        ],
    })

    out_path = out_dir / "feature_type_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="submission_diagnostics.csv",
        help="Path to submission_diagnostics.csv or feature CSV.",
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
        help="Top-ranked models used for top-k analysis.",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path)

    for col in ["id", "score"]:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column: {col}")

    group_df, feature_cols = build_feature_group_scores(df)

    print(f"Loaded {len(df)} rows from {features_path}")
    print("\nAvailable feature types:")
    for c in feature_cols:
        print(f"  - {c}")

    print("\nDominant feature type counts across all suspects:")
    print(group_df["best_feature_type"].value_counts().to_string())

    print(f"\nDominant feature type counts in top {args.top_k}:")
    print(
        group_df.sort_values("score", ascending=False)
        .head(args.top_k)["best_feature_type"]
        .value_counts()
        .to_string()
    )

    group_df.to_csv(out_dir / "feature_type_scores.csv", index=False)
    write_summary_csv(group_df, feature_cols, out_dir, args.top_k)

    plot_feature_type_distribution(group_df, out_dir, top_k=None)
    plot_feature_type_distribution(group_df, out_dir, top_k=args.top_k)
    plot_feature_type_boxplot(group_df, feature_cols, out_dir)
    plot_top_vs_rest(group_df, feature_cols, out_dir, args.top_k)
    plot_topk_heatmap(group_df, feature_cols, out_dir, args.top_k)


if __name__ == "__main__":
    main()