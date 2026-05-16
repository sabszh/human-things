from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

try:
    from human_things.metadata import (
        CONTROL_MODELS,
        FIG_BG,
        MODEL_ALIASES,
        GRID_COLOR,
        MODEL_COLORS as COLORS,
        MODEL_LABELS,
        MODEL_ORDER,
        MUTED_TEXT,
        TEXT_COLOR,
    )
    from human_things.paths import (
        BASELINE_OUTPUT_DIR,
        BENCHMARK_SUMMARY,
        BENCHMARK_SUMMARY_WITH_JOINT_MATRIX,
        BENCHMARK_SUMMARY_WITH_V3,
        FIGURES_DIR,
        HUMAN_V1_OUTPUT_DIR,
        HUMAN_V1_SHUFFLED_OUTPUT_DIR,
        HUMAN_V2_OUTPUT_DIR,
        HUMAN_V3_OUTPUT_DIR,
        JOINT_MATRIX_OUTPUT_DIR,
        JOINT_MATRIX_SHUFFLED_OUTPUT_DIR,
        PIPELINE_STORY_DRAWIO,
        TRIPLET_SATISFACTION_SUMMARY,
    )
    from human_things.utils import display_path, fail
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from human_things.metadata import (
        CONTROL_MODELS,
        FIG_BG,
        MODEL_ALIASES,
        GRID_COLOR,
        MODEL_COLORS as COLORS,
        MODEL_LABELS,
        MODEL_ORDER,
        MUTED_TEXT,
        TEXT_COLOR,
    )
    from human_things.paths import (
        BASELINE_OUTPUT_DIR,
        BENCHMARK_SUMMARY,
        BENCHMARK_SUMMARY_WITH_JOINT_MATRIX,
        BENCHMARK_SUMMARY_WITH_V3,
        FIGURES_DIR,
        HUMAN_V1_OUTPUT_DIR,
        HUMAN_V1_SHUFFLED_OUTPUT_DIR,
        HUMAN_V2_OUTPUT_DIR,
        HUMAN_V3_OUTPUT_DIR,
        JOINT_MATRIX_OUTPUT_DIR,
        JOINT_MATRIX_SHUFFLED_OUTPUT_DIR,
        PIPELINE_STORY_DRAWIO,
        TRIPLET_SATISFACTION_SUMMARY,
    )
    from human_things.utils import display_path, fail

DEFAULT_SUMMARY = BENCHMARK_SUMMARY_WITH_JOINT_MATRIX
FALLBACK_SUMMARIES = [BENCHMARK_SUMMARY_WITH_V3, BENCHMARK_SUMMARY]
TRIPLET_SUMMARY = TRIPLET_SATISFACTION_SUMMARY
OUTPUT_DIR = FIGURES_DIR

TRAINING_LOGS = {
    "baseline": BASELINE_OUTPUT_DIR / "training_log.csv",
    "fixed_prototype_triplets": HUMAN_V1_OUTPUT_DIR / "training_log.csv",
    "fixed_prototype_control": HUMAN_V1_SHUFFLED_OUTPUT_DIR / "training_log.csv",
    "batch_prototype_triplets": HUMAN_V2_OUTPUT_DIR / "training_log.csv",
    "high_pressure_triplets": HUMAN_V3_OUTPUT_DIR / "training_log.csv",
    "joint_matrix_alignment": JOINT_MATRIX_OUTPUT_DIR / "training_log.csv",
    "matrix_control": JOINT_MATRIX_SHUFFLED_OUTPUT_DIR / "training_log.csv",
}
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "paper_pastel",
    ["#fff7ed", "#edf4ff", "#e0f3da", "#eee2ff"],
)


def normalize_model_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "model" in frame.columns:
        frame["model"] = frame["model"].replace(MODEL_ALIASES)
    return frame


def sort_by_model_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    frame = frame.copy()
    frame["_order"] = frame["model"].map(order).fillna(999)
    return frame.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing benchmark summary: {path}. Run scripts/09_benchmark_embeddings.py first.")
    frame = pd.read_csv(path)
    if "model" not in frame.columns:
        fail(f"{path} is missing a model column.")
    return sort_by_model_order(normalize_model_column(frame))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "figure.facecolor": FIG_BG,
            "savefig.facecolor": FIG_BG,
            "axes.facecolor": FIG_BG,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.edgecolor": "none",
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": MUTED_TEXT,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
        }
    )


def labels_for(frame: pd.DataFrame) -> List[str]:
    return [MODEL_LABELS.get(model, model) for model in frame["model"]]


def colors_for(frame: pd.DataFrame) -> List[str]:
    return [COLORS.get(model, "#6B7280") for model in frame["model"]]


def is_control(model: str) -> bool:
    return model in CONTROL_MODELS


def label_for(model: str, single_line: bool = False) -> str:
    label = MODEL_LABELS.get(model, model)
    if single_line:
        label = label.replace("\n", " ")
    return label


def alpha_for(model: str) -> float:
    return 0.42 if is_control(model) else 0.96


def linewidth_for(model: str) -> float:
    return 1.5 if is_control(model) else 2.3


def marker_size_for(model: str) -> float:
    return 58 if is_control(model) else 92


def apply_control_bar_style(bars, models: pd.Series | List[str]) -> None:
    for bar, model in zip(bars, models):
        if is_control(str(model)):
            bar.set_alpha(0.45)
            bar.set_color(COLORS.get(str(model), "#B8C0CC"))


def style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(FIG_BG)
    for ax in fig.axes:
        ax.set_facecolor(FIG_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        if not ax.images:
            ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.35, linewidth=0.8)
            ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    style_figure(fig)
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": display_path(png), "svg": display_path(svg)}


def bar_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
    baseline_line: bool = True,
) -> Dict[str, str]:
    if metric not in frame.columns:
        fail(f"Missing metric column in summary: {metric}")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(frame))
    values = frame[metric].to_numpy(dtype=float)
    bars = ax.bar(x, values, color=colors_for(frame), width=0.72, edgecolor="none")
    apply_control_bar_style(bars, frame["model"])
    if baseline_line and "baseline" in set(frame["model"]):
        baseline_value = float(frame.loc[frame["model"] == "baseline", metric].iloc[0])
        ax.axhline(baseline_value, color="#111827", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.text(len(frame) - 0.55, baseline_value, "image-only", va="bottom", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for(frame))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lower = max(0.0, float(np.nanmin(values)) - 0.02)
    upper = min(1.0, float(np.nanmax(values)) + 0.02)
    if upper - lower < 0.08:
        pad = (0.08 - (upper - lower)) / 2
        lower = max(0.0, lower - pad)
        upper = min(1.0, upper + pad)
    ax.set_ylim(lower, upper)
    return save_figure(fig, output_dir, stem)


def grouped_metrics(
    frame: pd.DataFrame,
    metrics: List[str],
    labels: List[str],
    title: str,
    output_dir: Path,
    stem: str,
) -> Dict[str, str]:
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(frame))
    width = 0.24
    offsets = np.linspace(-width, width, len(metrics))
    palette = ["#1D4ED8", "#059669", "#DC2626", "#7C3AED"]
    for metric, label, offset, color in zip(metrics, labels, offsets, palette):
        bars = ax.bar(
            x + offset,
            frame[metric].to_numpy(dtype=float),
            width=width,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="none",
        )
        apply_control_bar_style(bars, frame["model"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for(frame))
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(ncols=min(3, len(metrics)), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_ylim(0, min(1.0, max(frame[metrics].max().max() + 0.04, 0.1)))
    return save_figure(fig, output_dir, stem)


def retrieval_curve_plot(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    metrics = ["image_retrieval_hit@1", "image_retrieval_hit@5", "image_retrieval_hit@10"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.array([1, 5, 10])
    for _, row in frame.iterrows():
        model = row["model"]
        y = row[metrics].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth_for(model),
            markersize=5 if is_control(model) else 6.5,
            color=COLORS.get(model, "#6B7280"),
            alpha=alpha_for(model),
            label=label_for(model, single_line=True),
        )
    ax.set_xticks(x)
    ax.set_xlabel("retrieval k")
    ax.set_ylabel("hit rate")
    ax.set_title("Retrieval Curves Across Models")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_ylim(0.68, 0.96)
    return save_figure(fig, output_dir, "figure_retrieval_curves")


def classification_retrieval_scatter(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    required = ["test_top1", "image_retrieval_hit@1"]
    missing = [metric for metric in required if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for _, row in frame.iterrows():
        model = row["model"]
        ax.scatter(
            row["test_top1"],
            row["image_retrieval_hit@1"],
            s=marker_size_for(model),
            color=COLORS.get(model, "#6B7280"),
            edgecolor="none",
            alpha=alpha_for(model),
            zorder=3,
        )
        ax.annotate(
            label_for(model, single_line=True),
            (row["test_top1"], row["image_retrieval_hit@1"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=MUTED_TEXT,
        )
    ax.set_xlabel("test top-1 accuracy")
    ax.set_ylabel("image retrieval hit@1")
    ax.set_title("Classification and Retrieval Move Together")
    return save_figure(fig, output_dir, "figure_classification_vs_retrieval")


def semantic_vs_human_scatter(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    required = [
        "human_similarity_pair_spearman",
        "nameability_mean_spearman",
        "object_properties_mean_spearman",
    ]
    missing = [metric for metric in required if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharex=True)
    panels = [
        ("nameability_mean_spearman", "nameability Spearman"),
        ("object_properties_mean_spearman", "object-property Spearman"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        for _, row in frame.iterrows():
            model = row["model"]
            ax.scatter(
                row["human_similarity_pair_spearman"],
                row[metric],
                s=marker_size_for(model),
                color=COLORS.get(model, "#6B7280"),
                edgecolor="none",
                alpha=alpha_for(model),
                zorder=3,
            )
            ax.annotate(
                label_for(model, single_line=True),
                (row["human_similarity_pair_spearman"], row[metric]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=MUTED_TEXT,
            )
        ax.set_xlabel("human-pair Spearman")
        ax.set_ylabel(ylabel)
    fig.suptitle("Does Human-Source Alignment Transfer to THINGSplus Variables?", y=1.02, fontsize=12)
    return save_figure(fig, output_dir, "figure_semantic_transfer_vs_human_alignment")


def benchmark_scorecard(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    metrics = [
        "test_top1",
        "image_retrieval_hit@1",
        "image_to_concept_hit@1",
        "category_linear_probe_53",
        "human_similarity_pair_spearman",
        "object_properties_mean_spearman",
    ]
    labels = ["top-1", "retrieval@1", "concept@1", "category", "human rho", "properties"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    values = frame[metrics].to_numpy(dtype=float)
    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    scaled = (values - col_min) / np.maximum(col_max - col_min, 1e-9)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(scaled, cmap=PASTEL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels([label_for(model, single_line=True) for model in frame["model"]])
    ax.set_title("Benchmark Scorecard Within Each Metric")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8, color=TEXT_COLOR)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("within-metric relative score")
    return save_figure(fig, output_dir, "figure_benchmark_scorecard")


def benchmark_rank_bump_chart(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    metrics = [
        "test_top1",
        "image_retrieval_hit@1",
        "human_similarity_pair_spearman",
        "category_linear_probe_53",
        "object_properties_mean_spearman",
    ]
    labels = ["top-1", "retrieval@1", "human rho", "category", "properties"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    ranks = frame[metrics].rank(axis=0, method="min", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    x = np.arange(len(metrics))
    for idx, row in frame.iterrows():
        model = row["model"]
        y = ranks.iloc[idx].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth_for(model),
            markersize=5 if is_control(model) else 6,
            color=COLORS.get(model, "#6B7280"),
            alpha=alpha_for(model),
            label=label_for(model, single_line=True),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("rank within metric")
    ax.set_title("Model Ranks Depend on Which Benchmark Is Valued")
    ax.set_ylim(len(frame) + 0.4, 0.6)
    ax.set_yticks(np.arange(1, len(frame) + 1))
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return save_figure(fig, output_dir, "figure_benchmark_rank_bump_chart")


def tradeoff_plot(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    required = ["image_retrieval_hit@1", "human_similarity_pair_spearman"]
    missing = [metric for metric in required if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for _, row in frame.iterrows():
        model = row["model"]
        ax.scatter(
            row["image_retrieval_hit@1"],
            row["human_similarity_pair_spearman"],
            s=marker_size_for(model),
            color=COLORS.get(model, "#6B7280"),
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha_for(model),
            zorder=3,
        )
        ax.annotate(
            label_for(model, single_line=True),
            (row["image_retrieval_hit@1"], row["human_similarity_pair_spearman"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("image retrieval hit@1")
    ax.set_ylabel("human-similarity pair Spearman")
    ax.set_title("Tradeoff Between Practical Retrieval and Human-Source Alignment")
    return save_figure(fig, output_dir, "figure_tradeoff_retrieval_vs_human_alignment")


def delta_heatmap(frame: pd.DataFrame, metrics: List[str], output_dir: Path) -> Dict[str, str]:
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")
    if "baseline" not in set(frame["model"]):
        fail("Cannot make delta heatmap without a baseline row.")
    baseline = frame.set_index("model").loc["baseline", metrics].astype(float)
    delta = frame.set_index("model")[metrics].astype(float) - baseline
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    vmax = float(np.nanmax(np.abs(delta.to_numpy())))
    vmax = max(vmax, 1e-4)
    im = ax.imshow(delta.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        [
            "top-1",
            "retrieval@1",
            "human rho",
            "category",
            "nameability",
            "properties",
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(delta.index)))
    ax.set_yticklabels([label_for(model, single_line=True) for model in delta.index])
    ax.set_title("Metric Deltas Relative to Image-Only Baseline")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("delta vs baseline")
    return save_figure(fig, output_dir, "figure_delta_heatmap_vs_baseline")


def metric_delta_profile(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    metrics = [
        "test_top1",
        "image_retrieval_hit@1",
        "human_similarity_pair_spearman",
        "category_linear_probe_53",
        "object_properties_mean_spearman",
    ]
    labels = ["top-1", "retrieval@1", "human rho", "category probe", "properties"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")
    if "baseline" not in set(frame["model"]):
        fail("Cannot make metric delta profile without a baseline row.")

    baseline = frame.set_index("model").loc["baseline", metrics].astype(float)
    delta = frame.set_index("model")[metrics].astype(float) - baseline
    plot_delta = delta.drop(index="baseline", errors="ignore")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(metrics))
    ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.72)
    for model, row in plot_delta.iterrows():
        ax.plot(
            x,
            row.to_numpy(dtype=float),
            marker="o",
            linewidth=linewidth_for(model),
            markersize=4.8 if is_control(model) else 5.8,
            color=COLORS.get(model, "#6B7280"),
            alpha=alpha_for(model),
            label=label_for(model, single_line=True),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("delta vs image-only baseline")
    ax.set_title("How Each Human-Informed Variant Moves the Embedding")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return save_figure(fig, output_dir, "figure_metric_delta_profiles")



def joint_matrix_control_delta_plot(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str] | None:
    required_models = {"baseline", "joint_matrix_alignment", "matrix_control"}
    if not required_models.issubset(set(frame["model"])):
        print("Skipping joint matrix control delta figure; missing baseline/joint rows.")
        return None
    metrics = [
        "test_top1",
        "image_retrieval_hit@1",
        "human_similarity_pair_spearman",
        "nameability_mean_spearman",
        "object_properties_mean_spearman",
    ]
    labels = ["top-1", "retrieval@1", "human rho", "nameability", "properties"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    indexed = frame.set_index("model")
    baseline = indexed.loc["baseline", metrics].astype(float)
    human = indexed.loc["joint_matrix_alignment", metrics].astype(float) - baseline
    shuffled = indexed.loc["matrix_control", metrics].astype(float) - baseline

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y = np.arange(len(metrics))
    for i in y:
        ax.plot([shuffled.iloc[i], human.iloc[i]], [i, i], color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.scatter(
        shuffled.to_numpy(dtype=float),
        y,
        color=COLORS["matrix_control"],
        s=85,
        alpha=alpha_for("matrix_control"),
        label=label_for("matrix_control", single_line=True),
        zorder=2,
    )
    ax.scatter(
        human.to_numpy(dtype=float),
        y,
        color=COLORS["joint_matrix_alignment"],
        s=85,
        label=label_for("joint_matrix_alignment", single_line=True),
        zorder=3,
    )
    ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta vs image-only baseline")
    ax.set_title("Matrix Alignment Strategy vs Matched Control")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    return save_figure(fig, output_dir, "figure_joint_matrix_vs_shuffled_deltas")

def fixed_prototype_control_delta_plot(frame: pd.DataFrame, output_dir: Path) -> Dict[str, str] | None:
    required_models = {"baseline", "fixed_prototype_triplets", "fixed_prototype_control"}
    if not required_models.issubset(set(frame["model"])):
        print("Skipping fixed-prototype control delta figure; missing baseline/fixed-prototype rows.")
        return None
    metrics = [
        "test_top1",
        "image_retrieval_hit@1",
        "human_similarity_pair_spearman",
        "nameability_mean_spearman",
        "object_properties_mean_spearman",
    ]
    labels = ["top-1", "retrieval@1", "human rho", "nameability", "properties"]
    missing = [metric for metric in metrics if metric not in frame.columns]
    if missing:
        fail(f"Missing metric columns in summary: {missing}")

    indexed = frame.set_index("model")
    baseline = indexed.loc["baseline", metrics].astype(float)
    human = indexed.loc["fixed_prototype_triplets", metrics].astype(float) - baseline
    shuffled = indexed.loc["fixed_prototype_control", metrics].astype(float) - baseline

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y = np.arange(len(metrics))
    for i in y:
        ax.plot([shuffled.iloc[i], human.iloc[i]], [i, i], color="#CBD5E1", linewidth=3.0, zorder=1)
    ax.scatter(
        shuffled.to_numpy(dtype=float),
        y,
        color=COLORS["fixed_prototype_control"],
        alpha=alpha_for("fixed_prototype_control"),
        s=80,
        label=label_for("fixed_prototype_control", single_line=True),
        zorder=2,
    )
    ax.scatter(
        human.to_numpy(dtype=float),
        y,
        color=COLORS["fixed_prototype_triplets"],
        s=80,
        label=label_for("fixed_prototype_triplets", single_line=True),
        zorder=3,
    )
    ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta vs image-only baseline")
    ax.set_title("Fixed-Prototype Triplets vs Matched Control")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    return save_figure(fig, output_dir, "figure_fixed_prototype_triplets_vs_control")


def triplet_satisfaction_plot(path: Path, output_dir: Path) -> Dict[str, str] | None:
    if not path.exists():
        print(f"Skipping triplet satisfaction figure; missing {path}")
        return None
    frame = normalize_model_column(pd.read_csv(path))
    required = {"model", "triplet_set", "satisfaction_rate", "margin_satisfaction_rate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")

    real = frame[frame["triplet_set"] == "real_train_triplets"].copy()
    shuffled = frame[frame["triplet_set"] == "shuffled_train_triplets"].copy()
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    real["_order"] = real["model"].map(order).fillna(999)
    shuffled["_order"] = shuffled["model"].map(order).fillna(999)
    real = real.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)
    shuffled = shuffled.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)
    if list(real["model"]) != list(shuffled["model"]):
        fail("Triplet satisfaction real/shuffled rows do not contain the same models in the same order.")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(real))
    width = 0.26
    ax.bar(
        x - width,
        real["satisfaction_rate"].to_numpy(dtype=float),
        width=width,
        label="real triplets: sim(pos) > sim(neg)",
        color=colors_for(real),
        alpha=0.92,
    )
    ax.bar(
        x,
        real["margin_satisfaction_rate"].to_numpy(dtype=float),
        width=width,
        label="real triplets: margin >= 0.2",
        color="#111827",
        alpha=0.72,
    )
    ax.bar(
        x + width,
        shuffled["satisfaction_rate"].to_numpy(dtype=float),
        width=width,
        label="shuffled triplets",
        color="#D1D5DB",
        edgecolor="#9CA3AF",
        alpha=0.95,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for(real))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("triplet satisfaction rate")
    ax.set_title("Human Triplet Satisfaction in Concept Embedding Space")
    ax.legend(ncols=1, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return save_figure(fig, output_dir, "figure_triplet_satisfaction")


def triplet_margin_interval_plot(path: Path, output_dir: Path) -> Dict[str, str] | None:
    if not path.exists():
        print(f"Skipping triplet margin interval figure; missing {path}")
        return None
    frame = normalize_model_column(pd.read_csv(path))
    required = {
        "model",
        "triplet_set",
        "p05_similarity_margin",
        "median_similarity_margin",
        "p95_similarity_margin",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")

    real = frame[frame["triplet_set"] == "real_train_triplets"].copy()
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    real["_order"] = real["model"].map(order).fillna(999)
    real = real.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)
    if real.empty:
        print("Skipping triplet margin interval figure; no real triplet rows.")
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(real))
    low = real["p05_similarity_margin"].to_numpy(dtype=float)
    med = real["median_similarity_margin"].to_numpy(dtype=float)
    high = real["p95_similarity_margin"].to_numpy(dtype=float)
    for i, row in real.iterrows():
        model = row["model"]
        color = COLORS.get(model, "#6B7280")
        ax.plot(
            [low[i], high[i]],
            [i, i],
            color=color,
            linewidth=4 if is_control(model) else 5,
            alpha=0.18 if is_control(model) else 0.28,
            solid_capstyle="round",
        )
        ax.scatter(med[i], i, color=color, s=58 if is_control(model) else 85, alpha=alpha_for(model), zorder=3)
    ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.7)
    ax.axvline(0.2, color="#111827", linewidth=1.0, linestyle="--", alpha=0.45)
    ax.text(0.202, -0.48, "margin 0.2", fontsize=8, color="#374151", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(model, single_line=True) for model in real["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("model cosine margin: sim(anchor,pos) - sim(anchor,neg)")
    ax.set_title("Distribution of Human-Triplet Margins in Embedding Space")
    return save_figure(fig, output_dir, "figure_triplet_margin_intervals")


def triplet_real_vs_shuffled_gap_plot(path: Path, output_dir: Path) -> Dict[str, str] | None:
    if not path.exists():
        print(f"Skipping real-vs-shuffled triplet gap figure; missing {path}")
        return None
    frame = normalize_model_column(pd.read_csv(path))
    required = {"model", "triplet_set", "satisfaction_rate", "mean_similarity_margin"}
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")

    pivot = frame.pivot(index="model", columns="triplet_set", values=["satisfaction_rate", "mean_similarity_margin"])
    needed = [
        ("satisfaction_rate", "real_train_triplets"),
        ("satisfaction_rate", "shuffled_train_triplets"),
        ("mean_similarity_margin", "real_train_triplets"),
        ("mean_similarity_margin", "shuffled_train_triplets"),
    ]
    if any(col not in pivot.columns for col in needed):
        print("Skipping real-vs-shuffled triplet gap figure; missing real or shuffled rows.")
        return None

    out = pd.DataFrame(index=pivot.index)
    out["satisfaction_gap"] = (
        pivot[("satisfaction_rate", "real_train_triplets")]
        - pivot[("satisfaction_rate", "shuffled_train_triplets")]
    )
    out["margin_gap"] = (
        pivot[("mean_similarity_margin", "real_train_triplets")]
        - pivot[("mean_similarity_margin", "shuffled_train_triplets")]
    )
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    out["_order"] = out.index.map(lambda model: order.get(model, 999))
    out = out.sort_values(["_order"]).drop(columns=["_order"])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for model, row in out.iterrows():
        ax.scatter(
            row["satisfaction_gap"],
            row["margin_gap"],
            s=marker_size_for(model),
            color=COLORS.get(model, "#6B7280"),
            edgecolor="none",
            alpha=alpha_for(model),
            zorder=3,
        )
        ax.annotate(
            label_for(model, single_line=True),
            (row["satisfaction_gap"], row["margin_gap"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=MUTED_TEXT,
        )
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.5)
    ax.axvline(0, color="#111827", linewidth=1, alpha=0.5)
    ax.set_xlabel("real - shuffled triplet satisfaction")
    ax.set_ylabel("real - shuffled mean margin")
    ax.set_title("Human Triplet Structure Above Shuffled Control")
    return save_figure(fig, output_dir, "figure_triplet_real_vs_shuffled_gap")


def training_curves_plot(output_dir: Path) -> Dict[str, str] | None:
    curves = []
    for model, path in TRAINING_LOGS.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "val_top1" not in frame.columns:
            continue
        frame = frame.copy()
        frame["model"] = model
        frame["step"] = np.arange(1, len(frame) + 1)
        curves.append(frame[["model", "step", "val_top1"]])
    if not curves:
        print("Skipping training curve figure; no training logs found.")
        return None

    combined = pd.concat(curves, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model, group in combined.groupby("model", sort=False):
        ax.plot(
            group["step"],
            group["val_top1"],
            marker="o",
            linewidth=linewidth_for(model),
            markersize=3.8 if is_control(model) else 4.8,
            color=COLORS.get(model, "#6B7280"),
            alpha=alpha_for(model),
            label=label_for(model, single_line=True),
        )
    ax.set_xlabel("training checkpoint / epoch")
    ax.set_ylabel("validation top-1 accuracy")
    ax.set_title("Validation Accuracy Across Training Runs")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return save_figure(fig, output_dir, "figure_training_curves_val_top1")


def write_caption_notes(frame: pd.DataFrame, output_dir: Path, figure_paths: Dict[str, Dict[str, str]]) -> Path:
    notes = {
        "status": "ok",
        "source_models": frame["model"].tolist(),
        "figures": figure_paths,
        "interpretation_notes": [
            "Fixed-prototype triplets and their shuffled control show nearly identical gains on classification and retrieval, suggesting generic fine-tuning or regularization rather than human-specific structure.",
            "High-pressure triplets increase within-source human-similarity alignment, but this is a manipulation check because the model is trained from the same human-similarity source.",
            "Triplet satisfaction shows whether margin-based human constraints are already present in the image-only embedding space.",
            "Joint matrix models test a different injection strategy: global relational alignment during THINGS classification training, with a shuffled-matrix control.",
            "External THINGSplus transfer is evaluated separately from human-source alignment to avoid treating within-source human similarity as an independent semantic benchmark.",
        ],
    }
    path = output_dir / "paper_figure_notes.json"
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from embedding benchmark summaries.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--triplet-summary-csv", type=Path, default=TRIPLET_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    summary_path = args.summary_csv.expanduser().resolve()
    if not summary_path.exists() and args.summary_csv == DEFAULT_SUMMARY:
        for fallback in FALLBACK_SUMMARIES:
            if fallback.exists():
                summary_path = fallback
                print(f"Default joint benchmark summary not found; using fallback {display_path(summary_path)}")
                break
    output_dir = args.output_dir.expanduser().resolve()
    setup_style()
    frame = load_summary(summary_path)

    figure_paths = {
        "classification_top1": bar_metric(
            frame,
            "test_top1",
            "test top-1 accuracy",
            "Concept Classification Accuracy",
            output_dir,
            "figure_classification_top1",
        ),
        "image_retrieval": grouped_metrics(
            frame,
            ["image_retrieval_hit@1", "image_retrieval_hit@5", "image_retrieval_hit@10"],
            ["hit@1", "hit@5", "hit@10"],
            "Image Retrieval Utility",
            output_dir,
            "figure_image_retrieval",
        ),
        "retrieval_curves": retrieval_curve_plot(frame, output_dir),
        "human_alignment": bar_metric(
            frame,
            "human_similarity_pair_spearman",
            "Spearman rho",
            "Within-Source Human Similarity Alignment",
            output_dir,
            "figure_human_similarity_alignment",
        ),
        "thingsplus_transfer": grouped_metrics(
            frame,
            ["nameability_mean_spearman", "lexical_concept_mean_spearman", "object_properties_mean_spearman"],
            ["nameability", "lexical/concept", "object properties"],
            "THINGSplus Transfer Benchmarks",
            output_dir,
            "figure_thingsplus_transfer",
        ),
        "classification_vs_retrieval": classification_retrieval_scatter(frame, output_dir),
        "semantic_transfer_vs_human_alignment": semantic_vs_human_scatter(frame, output_dir),
        "benchmark_scorecard": benchmark_scorecard(frame, output_dir),
        "benchmark_rank_bump_chart": benchmark_rank_bump_chart(frame, output_dir),
        "tradeoff": tradeoff_plot(frame, output_dir),
        "delta_heatmap": delta_heatmap(
            frame,
            [
                "test_top1",
                "image_retrieval_hit@1",
                "human_similarity_pair_spearman",
                "category_linear_probe_53",
                "nameability_mean_spearman",
                "object_properties_mean_spearman",
            ],
            output_dir,
        ),
        "metric_delta_profiles": metric_delta_profile(frame, output_dir),
    }
    fixed_prototype_control_paths = fixed_prototype_control_delta_plot(frame, output_dir)
    if fixed_prototype_control_paths is not None:
        figure_paths["fixed_prototype_triplets_vs_control"] = fixed_prototype_control_paths
    joint_control_paths = joint_matrix_control_delta_plot(frame, output_dir)
    if joint_control_paths is not None:
        figure_paths["joint_matrix_vs_shuffled_deltas"] = joint_control_paths
    triplet_paths = triplet_satisfaction_plot(args.triplet_summary_csv.expanduser().resolve(), output_dir)
    if triplet_paths is not None:
        figure_paths["triplet_satisfaction"] = triplet_paths
    triplet_margin_paths = triplet_margin_interval_plot(args.triplet_summary_csv.expanduser().resolve(), output_dir)
    if triplet_margin_paths is not None:
        figure_paths["triplet_margin_intervals"] = triplet_margin_paths
    triplet_gap_paths = triplet_real_vs_shuffled_gap_plot(args.triplet_summary_csv.expanduser().resolve(), output_dir)
    if triplet_gap_paths is not None:
        figure_paths["triplet_real_vs_shuffled_gap"] = triplet_gap_paths
    training_curve_paths = training_curves_plot(output_dir)
    if training_curve_paths is not None:
        figure_paths["training_curves_val_top1"] = training_curve_paths
    if PIPELINE_STORY_DRAWIO.exists():
        figure_paths["pipeline_story"] = {"drawio": display_path(PIPELINE_STORY_DRAWIO)}
    notes_path = write_caption_notes(frame, output_dir, figure_paths)

    print(f"Read: {summary_path}")
    for name, paths in figure_paths.items():
        if "png" in paths and "svg" in paths:
            print(f"{name}: {paths['png']} | {paths['svg']}")
        else:
            print(f"{name}: " + " | ".join(paths.values()))
    print(f"Wrote: {display_path(notes_path)}")


if __name__ == "__main__":
    main()
