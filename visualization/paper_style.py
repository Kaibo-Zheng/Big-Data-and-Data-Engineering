"""Shared foundation for the skill-styled ("scientific-paper-figure") report figures.

Bridges the external skill at F:\\Skills\\scientific-paper-figure (its
figure_primitives module) into this project, adapts it for Chinese labels, and
defines the f3 semantic palette mapping used by the paper-style builders.

Key adaptations vs. the raw skill:
- the skill's apply_rcparams uses Arial/DejaVu (no CJK); we prepend the project's
  detected CJK font so Chinese labels render;
- figures stay title-free ("图头" goes in the report caption); provenance lives in
  the .meta.json / .caption.md sidecars instead of an in-figure footnote.

Each figure is exported as PNG + PDF + SVG plus a .meta.json and a .caption.md via
``save_all``, matching the skill's output standard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Bridge the external skill -------------------------------------------- #
SKILL_DIR = Path(r"F:\Skills\scientific-paper-figure")
SKILL_SCRIPTS = SKILL_DIR / "scripts"
if SKILL_SCRIPTS.is_dir() and str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from figure_primitives import (  # noqa: E402  (skill module)
    add_visual_circle,
    apply_rcparams,
    embedding_block,
    f3_palette,
    junction,
    load_palette,
    route,
    save_figure,
    square_dx,
    tensor_strip,
    write_metadata,
)

from visualization.style import CHINESE_FONT  # noqa: E402

FIG_DIR = ROOT / "Illustration"
CSV_DIR = ROOT / "analysis" / "csv"

# --- f3 semantic palette --------------------------------------------------- #
F3 = f3_palette(load_palette())
INK = F3["structure"]            # #030303 — text / arrows / outlines
MUTED = "#6B7079"
BACKGROUND = F3["background"]

FROZEN_FILL = F3["frozen"]            # pale teal — frozen backbone
FROZEN_EDGE = "#90adb2"
TRAINABLE_FILL = F3["trainable"]      # cream — trainable modules
TRAINABLE_EDGE = F3["trainable_outline"]  # muted gold
HEAD_FILL = "#f6e2e2"                 # pale pink — classification head
HEAD_EDGE = F3["pink_accent"]
NEUTRAL_FILL = "#f4f6f8"

# Three detection classes (label) — distinct muted hues.
CLASS_F3 = {"human": F3["query_accent"], "machine": F3["pink_accent"], "hybrid": F3["graph_evidence"]}
# Pale fills (for big bars / pills that carry text alongside, not on top).
CLASS_FILL = {"human": "#d6e2f1", "machine": "#f6dada", "hybrid": "#d8e6df"}
SPLIT_F3 = {"train": F3["query_accent"], "test_a": TRAINABLE_EDGE}

DETECT_F3 = F3["query_accent"]   # blue
SOURCE_F3 = F3["pink_accent"]    # pink
OFFICIAL_F3 = TRAINABLE_EDGE     # gold
INIT_F3 = "#c9bfa0"              # muted cream-gray for the "initial" bar

# Eight machine-source families — muted, mutually distinguishable.
FAMILY_F3 = {
    0: "#598bce",  # OpenAI   — blue
    1: "#54a79c",  # Alibaba  — teal
    2: "#c2ae74",  # DeepSeek — gold
    3: "#9590ab",  # ByteDance— lavender
    4: "#ec99a4",  # Moonshot — pink
    5: "#6f8f7a",  # Google   — sage
    6: "#b08968",  # Anthropic— tan
    7: "#7f8794",  # xAI      — slate
}

LABEL_ZH = {"human": "人类撰写", "machine": "机器生成", "hybrid": "人机协作"}
SPLIT_ZH = {"train": "训练集", "test_a": "测试集 A"}
FAMILY_LABELS = {
    0: "OpenAI", 1: "Alibaba", 2: "DeepSeek", 3: "ByteDance",
    4: "Moonshot", 5: "Google", 6: "Anthropic", 7: "xAI",
}


def apply_paper_style(base_font_size: float = 9.5) -> None:
    """Apply the skill's paper rcParams, then layer in CJK font support."""
    apply_rcparams(base_font_size)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [CHINESE_FONT, "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def write_caption(prefix: str, title: str, description: str, provenance: str,
                  output_dir: "str | Path" = FIG_DIR) -> Path:
    """Write a concise Markdown caption sidecar (Chinese), skill-style."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{prefix}.caption.md"
    text = f"# {title}\n\n{description}\n\n_数据来源 / Provenance: {provenance}_\n"
    out.write_text(text, encoding="utf-8")
    return out


def save_all(fig, prefix: str, *, title: str, description: str, provenance: str,
             meta: "dict | None" = None, formats: Sequence[str] = ("png", "pdf", "svg"),
             dpi: int = 300, output_dir: "str | Path" = FIG_DIR) -> dict:
    """Save fig as png/pdf/svg + write a .meta.json and a .caption.md sidecar.

    Returns the format->Path map from the skill's save_figure.
    """
    paths = save_figure(fig, output_dir, prefix, formats=tuple(formats), dpi=dpi)
    meta = dict(meta or {})
    meta.setdefault("style", "scientific-paper-figure skill (f3 palette)")
    meta.setdefault("provenance", provenance)
    meta["outputs"] = {fmt: str(p) for fmt, p in paths.items()}
    write_metadata(output_dir, prefix, meta)
    write_caption(prefix, title, description, provenance, output_dir=output_dir)
    return paths


__all__ = [
    "FIG_DIR", "CSV_DIR", "F3", "INK", "MUTED", "BACKGROUND",
    "FROZEN_FILL", "FROZEN_EDGE", "TRAINABLE_FILL", "TRAINABLE_EDGE",
    "HEAD_FILL", "HEAD_EDGE", "NEUTRAL_FILL",
    "CLASS_F3", "CLASS_FILL", "SPLIT_F3", "DETECT_F3", "SOURCE_F3", "OFFICIAL_F3", "INIT_F3",
    "FAMILY_F3", "LABEL_ZH", "SPLIT_ZH", "FAMILY_LABELS",
    "apply_paper_style", "save_all", "write_caption",
    # re-exported skill primitives
    "route", "junction", "embedding_block", "tensor_strip", "square_dx", "add_visual_circle",
]
