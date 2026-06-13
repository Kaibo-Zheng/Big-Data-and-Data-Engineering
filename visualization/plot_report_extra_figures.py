"""Generate additional code-drawn metric figures for template.tex."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.paper_style import (  # noqa: E402
    DETECT_F3,
    FIG_DIR,
    INK,
    OFFICIAL_F3,
    SOURCE_F3,
    apply_paper_style,
    save_all,
)

CSV_DIR = ROOT / "result" / "csv"


def method_performance_comparison() -> None:
    df = pd.read_csv(CSV_DIR / "llm_text_method_comparison.csv")
    pivot = df.pivot_table(index=["order", "method_label"], columns="task", values="macro_f1").reset_index()
    pivot = pivot.sort_values("order")

    methods = pivot["method_label"].tolist()
    x = range(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_facecolor("white")
    bars = []
    for offset, key, label, color in [
        (-width, "detect", "检测 Macro-F1", DETECT_F3),
        (0.0, "source", "溯源 Macro-F1", SOURCE_F3),
        (width, "official", "加权估计分数", OFFICIAL_F3),
    ]:
        values = pivot[key].astype(float).to_numpy()
        bar = ax.bar([i + offset for i in x], values, width=width, label=label, color=color, alpha=0.95)
        bars.append(bar)
        for rect, value in zip(bar, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color=INK,
            )

    ax.set_xticks(list(x), methods)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("分数")
    ax.grid(axis="y", color="#d9dde3", lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=3)

    save_all(
        fig,
        "LLM_E3_method_performance_comparison",
        title="Method performance comparison",
        description="A grouped bar chart comparing detection Macro-F1, source-attribution Macro-F1, and estimated official score.",
        provenance="result/csv/llm_text_method_comparison.csv",
    )
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    method_performance_comparison()
    print("Wrote extra metric figure to Illustration/.")


if __name__ == "__main__":
    main()
