"""Shared plotting style for report-quality figures."""

from __future__ import annotations

import warnings

import matplotlib as mpl

if mpl.get_backend().lower() != "agg":
    mpl.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm


CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)


def _detect_cjk_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    warnings.warn(
        "No CJK font found; Chinese labels may not render correctly.",
        stacklevel=2,
    )
    return "DejaVu Sans"


CHINESE_FONT: str = _detect_cjk_font()

# Muted, print-friendly colors. Keep classes stable across all figures.
CLASS_PALETTE: dict[str, str] = {
    "human": "#4F6D8A",
    "machine": "#9A5B55",
    "hybrid": "#6F8662",
}

SPLIT_PALETTE: dict[str, str] = {
    "train": "#4F6D8A",
    "test_a": "#B38B59",
}

ACADEMIC_PALETTE: list[str] = [
    "#4F6D8A",
    "#9A5B55",
    "#6F8662",
    "#B38B59",
    "#6B6A88",
    "#5C7F7B",
    "#8A7A63",
    "#7A7F87",
]

# Machine source families. Index order matches ckpt/**/class_names.json and the
# family head's output dimension, so B3/D3/D4/D5 and the schematics stay aligned.
FAMILY_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
FAMILY_LABELS: dict[int, str] = {
    0: "OpenAI",
    1: "Alibaba",
    2: "DeepSeek",
    3: "ByteDance",
    4: "Moonshot",
    5: "Google",
    6: "Anthropic",
    7: "xAI",
}
FAMILY_PALETTE: dict[int, str] = {i: ACADEMIC_PALETTE[i] for i in FAMILY_ORDER}

NEUTRAL_TEXT = "#2F2F2F"
GRID_COLOR = "#D8DCE2"
VIZ = {
    "figure_size": (7.2, 4.8),
    "dpi": 300,
}


def apply_style() -> None:
    """Apply a restrained matplotlib/seaborn theme for academic reports."""
    sns.set_theme(style="ticks", palette=ACADEMIC_PALETTE)

    mpl.rcParams.update(
        {
            "font.sans-serif": [CHINESE_FONT, "DejaVu Sans", "Arial"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "figure.titlesize": 12,
            "figure.titleweight": "semibold",
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": NEUTRAL_TEXT,
            "xtick.color": NEUTRAL_TEXT,
            "ytick.color": NEUTRAL_TEXT,
            "text.color": NEUTRAL_TEXT,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "figure.figsize": list(VIZ["figure_size"]),
            "figure.dpi": 120,
            "savefig.dpi": VIZ["dpi"],
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.08,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
        }
    )


def add_source_note(fig: plt.Figure, source_csv: str) -> None:
    source = source_csv if "/" in source_csv or "\\" in source_csv else f"analysis/csv/{source_csv}"
    fig.text(
        0.01,
        0.006,
        f"Source: {source}",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#8A8A8A",
    )


def fmt_thousands(x: float, _: int = 0) -> str:
    return f"{int(x):,}"


def fmt_percent(x: float, _: int = 0) -> str:
    return f"{x * 100:.1f}%"


def fmt_compact(x: float, _: int = 0) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def save(fig: plt.Figure, out_path, *, source_csv: str | None = None) -> None:
    if source_csv:
        add_source_note(fig, source_csv)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    fig.savefig(out_path, dpi=VIZ["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)


__all__ = [
    "ACADEMIC_PALETTE",
    "CHINESE_FONT",
    "CLASS_PALETTE",
    "FAMILY_LABELS",
    "FAMILY_ORDER",
    "FAMILY_PALETTE",
    "SPLIT_PALETTE",
    "apply_style",
    "fmt_compact",
    "fmt_percent",
    "fmt_thousands",
    "save",
]
