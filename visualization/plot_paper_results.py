"""Generate the consolidated Group-D experimental result figure.

The report does not present a heavy method benchmark. Instead, this script
builds one multi-panel summary that shows final validation performance, family
training dynamics, and Test A prediction distributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.paper_style import (  # noqa: E402
    CLASS_F3,
    DETECT_F3,
    FAMILY_F3,
    FAMILY_LABELS,
    FIG_DIR,
    INK,
    MUTED,
    OFFICIAL_F3,
    SOURCE_F3,
    apply_paper_style,
    save_all,
)
from visualization.style import fmt_thousands  # noqa: E402


CSV_DIR = ROOT / "result" / "csv"
OUTPUT_PREFIX = "LLM_D1_experimental_summary"
STALE_PREFIXES = (
    "LLM_D0_baseline_confusion",
    "LLM_D1_method_comparison",
    "LLM_D2_label_performance",
    "LLM_D3_family_improvement",
    "LLM_D4_training_curve",
    "LLM_D5_submission_summary",
    OUTPUT_PREFIX,
)
LABEL_ORDER = ["human", "machine", "hybrid"]
LABEL_EN = {
    "human": "Human",
    "machine": "AI-generated",
    "hybrid": "Hybrid",
}
PANEL_LABEL_OFFSET = (-8, 8)
PANEL_LABEL_SIZE = 13.0


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name)


def clean_result_outputs() -> None:
    for prefix in STALE_PREFIXES:
        for suffix in (".png", ".pdf", ".svg", ".caption.md", ".meta.json"):
            (FIG_DIR / f"{prefix}{suffix}").unlink(missing_ok=True)


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#d9dde3", lw=0.7, alpha=0.9)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def add_panel_label(ax, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=PANEL_LABEL_OFFSET,
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=INK,
        clip_on=False,
        annotation_clip=False,
    )


def write_english_caption() -> None:
    caption = FIG_DIR / f"{OUTPUT_PREFIX}.caption.md"
    caption.write_text(
        "# Experimental Results Summary\n\n"
        "A four-panel overview of the final system: validation performance, "
        "source-attribution training dynamics, Test A label distribution, and "
        "Test A source-family distribution.\n\n"
        "_Provenance: result/csv/llm_text_model_metrics.csv; "
        "result/csv/llm_text_family_training_history.csv; "
        "result/csv/llm_text_submission_summary.csv_\n",
        encoding="utf-8",
    )


def panel_validation_performance(ax, metrics: pd.DataFrame) -> None:
    metrics = metrics.set_index("task")
    tasks = ["Detection", "Source attribution"]
    accuracy = [
        float(metrics.loc["label", "accuracy"]),
        float(metrics.loc["family", "accuracy"]),
    ]
    macro_f1 = [
        float(metrics.loc["label", "macro_f1"]),
        float(metrics.loc["family", "macro_f1"]),
    ]
    x = np.arange(len(tasks))
    width = 0.32

    bars_acc = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=DETECT_F3,
        alpha=0.94,
    )
    bars_f1 = ax.bar(
        x + width / 2,
        macro_f1,
        width,
        label="Macro-F1",
        color=SOURCE_F3,
        alpha=0.94,
    )

    ax.set_xticks(x, tasks)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    style_axis(ax)

    for bars, metric_name in ((bars_acc, "Acc"), (bars_f1, "F1")):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{metric_name}\n{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.9,
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
            )

    add_panel_label(ax, "a")


def panel_training_dynamics(ax, history: pd.DataFrame) -> None:
    history = history.sort_values("epoch")
    epochs = history["epoch"].to_numpy()

    ax.plot(
        epochs,
        history["family_valid_macro_f1"],
        marker="o",
        markersize=3.5,
        color=SOURCE_F3,
        linewidth=1.6,
        label="Valid Macro-F1",
    )
    ax.plot(
        epochs,
        history["family_valid_accuracy"],
        marker="s",
        markersize=3.1,
        color=DETECT_F3,
        linewidth=1.5,
        label="Valid accuracy",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0.62, 0.92)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax)

    ax_loss = ax.twinx()
    ax_loss.plot(
        epochs,
        history["train_loss"],
        marker="^",
        markersize=3.2,
        color=OFFICIAL_F3,
        linestyle="--",
        linewidth=1.4,
        label="Train loss",
    )
    ax_loss.set_ylabel("Train loss")
    ax_loss.set_ylim(0, float(history["train_loss"].max()) * 1.15)
    ax_loss.spines["top"].set_visible(False)

    best = history.loc[history["family_valid_macro_f1"].idxmax()]
    ax.scatter(
        [best["epoch"]],
        [best["family_valid_macro_f1"]],
        s=88,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.2,
        zorder=6,
    )
    ax.annotate(
        f"Best {best['family_valid_macro_f1']:.3f}\nEpoch {int(best['epoch'])}",
        xy=(best["epoch"], best["family_valid_macro_f1"]),
        xytext=(-2, -34),
        textcoords="offset points",
        fontsize=8.0,
        color=INK,
    )

    lines = ax.get_lines() + ax_loss.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", frameon=False, fontsize=8.1)
    add_panel_label(ax, "b")


def panel_test_label_distribution(ax, summary: pd.DataFrame) -> None:
    labels = summary[summary["group"] == "label"].copy()
    labels["name"] = pd.Categorical(labels["name"], LABEL_ORDER, ordered=True)
    labels = labels.sort_values("name")

    names = [LABEL_EN[str(name)] for name in labels["name"]]
    colors = [CLASS_F3[str(name)] for name in labels["name"]]
    bars = ax.bar(names, labels["count"], color=colors, width=0.58, alpha=0.94)

    ax.set_ylabel("Predicted samples")
    ax.set_ylim(0, float(labels["count"].max()) * 1.24)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    style_axis(ax)

    for bar, count, pct in zip(bars, labels["count"], labels["percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + float(labels["count"].max()) * 0.035,
            f"{int(count):,}\n{pct:.1%}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            color=INK,
        )
    add_panel_label(ax, "c")


def panel_test_family_distribution(ax, summary: pd.DataFrame) -> None:
    families = summary[summary["group"] == "family"].copy()
    families["code"] = families["code"].astype(int)
    families = families.sort_values("count", ascending=True)

    names = [FAMILY_LABELS[int(code)] for code in families["code"]]
    colors = [FAMILY_F3[int(code)] for code in families["code"]]
    bars = ax.barh(names, families["count"], color=colors, height=0.58, alpha=0.94)

    ax.set_xlabel("Predicted AI-generated samples")
    ax.set_xlim(0, float(families["count"].max()) * 1.28)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    style_axis(ax, grid_axis="x")

    max_count = float(families["count"].max())
    for bar, count, pct in zip(bars, families["count"], families["percent"]):
        ax.text(
            count + max_count * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count):,} ({pct:.1%})",
            ha="left",
            va="center",
            fontsize=8.0,
            color=INK,
        )
    add_panel_label(ax, "d")


def figD1_experimental_summary() -> None:
    metrics = load_csv("llm_text_model_metrics.csv")
    history = load_csv("llm_text_family_training_history.csv")
    summary = load_csv("llm_text_submission_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    panel_validation_performance(axes[0, 0], metrics)
    panel_training_dynamics(axes[0, 1], history)
    panel_test_label_distribution(axes[1, 0], summary)
    panel_test_family_distribution(axes[1, 1], summary)

    save_all(
        fig,
        OUTPUT_PREFIX,
        title="Experimental results summary",
        description=(
            "A four-panel summary of final validation metrics, source-attribution "
            "training dynamics, Test A label predictions, and Test A source-family predictions."
        ),
        provenance=(
            "result/csv/llm_text_model_metrics.csv; "
            "result/csv/llm_text_family_training_history.csv; "
            "result/csv/llm_text_submission_summary.csv"
        ),
    )
    write_english_caption()
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clean_result_outputs()
    figD1_experimental_summary()
    print("Wrote consolidated Group-D experimental result figure to Illustration/.")


if __name__ == "__main__":
    main()
