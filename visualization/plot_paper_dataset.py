"""Paper-style consolidated Group-B dataset figure for the course report.

Exports PNG/PDF/SVG plus .meta.json and .caption.md sidecars for the final
dataset-analysis figure. Reads only from analysis/csv/ and local logo assets.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.paper_style import (  # noqa: E402
    CLASS_F3,
    CSV_DIR,
    FAMILY_F3,
    FAMILY_LABELS,
    FIG_DIR,
    INK,
    SPLIT_F3,
    apply_paper_style,
    save_all,
)
from visualization.style import fmt_percent, fmt_thousands  # noqa: E402


LABEL_ORDER = ["human", "machine", "hybrid"]
LABEL_EN = {
    "human": "Human-written",
    "machine": "AI-generated",
    "hybrid": "Human-AI hybrid",
}
SPLIT_EN = {
    "train": "Train",
    "test_a": "Test A",
}
LOGO_FILES = {
    0: ROOT / "Illustration" / "logos" / "openai.png",
    1: ROOT / "Illustration" / "logos" / "alibaba.png",
    2: ROOT / "Illustration" / "logos" / "deepseek.png",
    3: ROOT / "Illustration" / "logos" / "bytedance.png",
    4: ROOT / "Illustration" / "logos" / "moonshot.png",
    5: ROOT / "Illustration" / "logos" / "google.png",
    6: ROOT / "Illustration" / "logos" / "anthropic.png",
    7: ROOT / "Illustration" / "logos" / "xai.png",
}
LOGO_CANVAS_SIZE = 256
LOGO_ZOOM = 0.13
LOGO_FIT_SIZE = {
    0: (92, 92),
    1: (96, 78),
    2: (138, 46),
    3: (96, 78),
    4: (92, 92),
    5: (138, 50),
    6: (90, 90),
    7: (88, 88),
}
PANEL_LABEL_OFFSET = (-8, 8)
PANEL_LABEL_SIZE = 13.0


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name)


def style_axis(ax, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color="#d9dde3", lw=0.7, alpha=0.9)
    other = "y" if grid_axis == "x" else "x"
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


def set_xlabel_aligned(ax, label: str, y: float = -0.18) -> None:
    ax.set_xlabel(label)
    ax.xaxis.set_label_coords(0.5, y)


def set_ylabel_aligned(ax, label: str, x: float = -0.16) -> None:
    ax.set_ylabel(label)
    ax.yaxis.set_label_coords(x, 0.5)


@lru_cache(maxsize=16)
def logo_image(family: int) -> Image.Image | None:
    path = LOGO_FILES.get(family)
    if not path or not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox:
        image = image.crop(bbox)

    max_w, max_h = LOGO_FIT_SIZE.get(family, (92, 92))
    scale = min(max_w / image.width, max_h / image.height)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (LOGO_CANVAS_SIZE, LOGO_CANVAS_SIZE), (255, 255, 255, 0))
    xy = ((LOGO_CANVAS_SIZE - new_size[0]) // 2, (LOGO_CANVAS_SIZE - new_size[1]) // 2)
    canvas.paste(image, xy, image)
    return canvas


def add_logo_badge(ax, family: int, x: float, y: float) -> None:
    image = logo_image(family)
    if image is None:
        return
    imagebox = OffsetImage(image, zoom=LOGO_ZOOM)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5), zorder=5)
    ax.add_artist(ab)


def figB0_dataset_overview() -> None:
    """Composite Group-B figure for the report body."""
    overview = load_csv("llm_text_overview.csv").copy()
    labels = load_csv("llm_text_label_distribution.csv").copy()
    families = load_csv("llm_text_family_distribution.csv").copy()
    length_hist = load_csv("llm_text_word_count_hist.csv").copy()
    methods = load_csv("llm_text_method_distribution.csv").copy()

    fig = plt.figure(figsize=(15.8, 8.7))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.55, 1.0, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.36,
        hspace=0.48,
    )
    ax_family = fig.add_subplot(gs[:, 0])
    ax_scale = fig.add_subplot(gs[0, 1])
    ax_label = fig.add_subplot(gs[0, 2])
    ax_length = fig.add_subplot(gs[1, 1])
    ax_hybrid = fig.add_subplot(gs[1, 2])

    # a. AI source-family distribution, kept as the main visual anchor.
    fam = families[families["label_name"] == "machine"].copy()
    fam["family"] = fam["family"].astype(int)
    fam = fam.sort_values("family")
    total_machine = fam["rows"].sum()
    y = list(range(len(fam)))
    bars = ax_family.barh(
        y,
        fam["rows"],
        color=[FAMILY_F3[int(f)] for f in fam["family"]],
        height=0.58,
        alpha=0.94,
    )
    ax_family.invert_yaxis()
    ax_family.set_yticks([])
    set_xlabel_aligned(ax_family, "AI-generated samples", y=-0.105)
    ax_family.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax_family.set_xlim(-1430, fam["rows"].max() * 1.22)
    ax_family.set_xticks([0, 1000, 2000, 3000, 4000])
    style_axis(ax_family, "x")
    add_panel_label(ax_family, "a")
    for yi, family, rows, bar in zip(y, fam["family"], fam["rows"], bars):
        badge = FancyBboxPatch(
            (-1370, yi - 0.33),
            1040,
            0.66,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#ffffff",
            edgecolor="#d8dde5",
            linewidth=0.9,
            zorder=2,
        )
        ax_family.add_patch(badge)
        add_logo_badge(ax_family, int(family), -1250, yi)
        ax_family.text(
            -1065,
            yi,
            FAMILY_LABELS[int(family)],
            ha="left",
            va="center",
            fontsize=9.1,
            fontweight="semibold",
            color=INK,
            zorder=5,
        )
        ax_family.text(
            rows + fam["rows"].max() * 0.022,
            bar.get_y() + bar.get_height() / 2,
            f"{rows:,.0f} ({rows / total_machine:.1%})",
            va="center",
            fontsize=8.6,
            color=INK,
        )

    # b. Dataset scale. Use horizontal bars so the axis grammar matches panels c/e.
    overview["split_name"] = overview["dataset"].map(SPLIT_EN)
    overview = overview.sort_values("rows", ascending=True)
    bars = ax_scale.barh(
        overview["split_name"],
        overview["rows"],
        color=[SPLIT_F3[x] for x in overview["dataset"]],
        height=0.50,
    )
    set_xlabel_aligned(ax_scale, "Samples", y=-0.145)
    ax_scale.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax_scale.set_xlim(0, overview["rows"].max() * 1.33)
    style_axis(ax_scale, "x")
    add_panel_label(ax_scale, "b")
    for bar, rows, size_mb in zip(bars, overview["rows"], overview["file_size_mb"]):
        ax_scale.text(
            rows + overview["rows"].max() * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{int(rows):,}\n{size_mb:.1f} MB",
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK,
            linespacing=1.15,
        )

    # c. Detection-label distribution.
    labels["label_name"] = pd.Categorical(labels["label_name"], LABEL_ORDER, ordered=True)
    labels = labels.sort_values("label_name")
    label_names = [LABEL_EN[x] for x in labels["label_name"].astype(str)]
    bars = ax_label.barh(
        label_names,
        labels["rows"],
        color=[CLASS_F3[x] for x in labels["label_name"].astype(str)],
        height=0.5,
    )
    ax_label.invert_yaxis()
    set_xlabel_aligned(ax_label, "Samples", y=-0.145)
    ax_label.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax_label.set_xlim(0, labels["rows"].max() * 1.35)
    style_axis(ax_label, "x")
    add_panel_label(ax_label, "c")
    for bar, rows, pct in zip(bars, labels["rows"], labels["percent"]):
        ax_label.text(
            rows + labels["rows"].max() * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{rows:,.0f}\n{pct:.1%}",
            va="center",
            fontsize=8.2,
            color=INK,
            linespacing=1.1,
        )

    # d. Text-length distribution.
    word_order = length_hist["word_bin"].drop_duplicates().tolist()
    short_bins = [
        item.replace("[", "").replace(")", "").replace(", ", "-")
        for item in word_order
    ]
    x = range(len(word_order))
    for dataset in ["train", "test_a"]:
        part = length_hist[length_hist["dataset"] == dataset].set_index("word_bin").loc[word_order]
        ax_length.plot(
            x,
            part["percent"],
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            color=SPLIT_F3[dataset],
            label=SPLIT_EN[dataset],
        )
    ax_length.set_xticks(list(x), short_bins, rotation=34, ha="right")
    set_ylabel_aligned(ax_length, "Sample share", x=-0.20)
    set_xlabel_aligned(ax_length, "Word-count bin", y=-0.305)
    ax_length.yaxis.set_major_formatter(FuncFormatter(fmt_percent))
    style_axis(ax_length, "y")
    ax_length.legend(loc="upper right", framealpha=0, handlelength=1.6)
    add_panel_label(ax_length, "d")

    # e. Hybrid construction methods.
    hybrid = methods[methods["label_name"] == "hybrid"].copy()
    short_method = {
        "machine-modify-human": "AI rewrite",
        "human-mix-machine": "Human-AI mix",
        "machine-continue-human (short prefix)": "Continue\nshort prefix",
        "machine-continue-human (long prefix)": "Continue\nlong prefix",
    }
    hybrid["method_name"] = hybrid["method"].map(short_method).fillna(hybrid["method"])
    hybrid = hybrid.sort_values("rows", ascending=True)
    bars = ax_hybrid.barh(hybrid["method_name"], hybrid["rows"], color=CLASS_F3["hybrid"], height=0.50)
    set_xlabel_aligned(ax_hybrid, "Hybrid samples", y=-0.305)
    ax_hybrid.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax_hybrid.set_xlim(0, hybrid["rows"].max() * 1.34)
    style_axis(ax_hybrid, "x")
    add_panel_label(ax_hybrid, "e")
    for bar, rows, pct in zip(bars, hybrid["rows"], hybrid["percent_in_label"]):
        ax_hybrid.text(
            rows + hybrid["rows"].max() * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{rows:,.0f}\n{pct:.1%}",
            va="center",
            fontsize=8.2,
            color=INK,
            linespacing=1.1,
        )

    sns.despine(fig=fig, left=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.105)
    save_all(
        fig,
        "LLM_B0_dataset_overview",
        title="Dataset overview and source statistics",
        description=(
            "Composite dataset-analysis figure: AI source-family balance, dataset scale, "
            "detection-label distribution, text-length distribution, and hybrid construction methods."
        ),
        provenance=(
            "analysis/csv/llm_text_overview.csv; analysis/csv/llm_text_label_distribution.csv; "
            "analysis/csv/llm_text_family_distribution.csv; analysis/csv/llm_text_word_count_hist.csv; "
            "analysis/csv/llm_text_method_distribution.csv; Illustration/logos/logo_sources.md"
        ),
    )
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figB0_dataset_overview()
    print("Wrote consolidated Group-B dataset overview figure to Illustration/")


if __name__ == "__main__":
    main()
