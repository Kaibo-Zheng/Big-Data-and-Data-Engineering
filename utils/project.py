"""Shared paths and label metadata for this repository."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TEST_JSONL = DATA_DIR / "test_a_release.jsonl"
REPORT_DIR = ROOT / "report"
ANALYSIS_DIR = ROOT / "analysis"
CSV_DIR = ANALYSIS_DIR / "csv"
FIGURE_DIR = ROOT / "visualization" / "figures"
RESULT_DIR = ROOT / "result"
RESULT_CSV_DIR = RESULT_DIR / "csv"
SUBMISSION_DIR = RESULT_DIR / "submissions"
CKPT_DIR = ROOT / "ckpt"
MODEL_DIR = ROOT / "model"
PRETRAINED_MODEL_DIR = MODEL_DIR

LABEL_NAMES = {0: "human", 1: "machine", 2: "hybrid"}
FAMILY_NAMES = {
    0: "openai",
    1: "alibaba",
    2: "deepseek",
    3: "bytedance",
    4: "moonshot",
    5: "google",
    6: "anthropic",
    7: "xai",
}
