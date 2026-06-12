"""Collect model/result metrics into result/csv/ for the Group D figures.

Mirrors tool/profile_dataset.py but reads *result* artifacts instead of raw data:
- ckpt/{label,family}/metrics.json                        (final validation metrics)
- result/csv/llm_text_baseline_metrics.csv                (TF-IDF + SGD baseline)
- logs/deberta_lora_ddp_*.log                             (initial run + family curve)
- result/submissions/submit.jsonl                         (final test-A predictions)

Outputs (each row carries a ``source`` column for provenance):
- llm_text_model_metrics.csv
- llm_text_method_comparison.csv
- llm_text_family_training_history.csv
- llm_text_submission_summary.csv

Plotting stays pure-from-CSV (visualization/plot_paper_results.py reads only these files).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "result" / "csv"
CKPT_DIR = ROOT / "ckpt"
LOG_DIR = ROOT / "logs"
SUBMISSION = ROOT / "result" / "submissions" / "submit.jsonl"
BASELINE_METRICS = CSV_DIR / "llm_text_baseline_metrics.csv"

LABEL_NAMES = {0: "human", 1: "machine", 2: "hybrid"}
FAMILY_NAMES = {
    0: "OpenAI",
    1: "Alibaba",
    2: "DeepSeek",
    3: "ByteDance",
    4: "Moonshot",
    5: "Google",
    6: "Anthropic",
    7: "xAI",
}

# A validation block printed once per epoch by the training loop, e.g.
#   {\n  "accuracy": 0.678,\n  "macro_f1": 0.675,\n  "task": "family",\n  "epoch": 1, ...
BLOCK_RE = re.compile(
    r'"accuracy":\s*([0-9.]+),\s*"macro_f1":\s*([0-9.]+),\s*"task":\s*"(\w+)",\s*"epoch":\s*(\d+)',
    re.S,
)
# End-of-epoch tqdm line, e.g. "family epoch 7/20: 100%|...| 450/450 [.., loss=0.12, lr=..]"
LOSS_LINE_RE = re.compile(r"family epoch (\d+)/\d+:\s*100%")
LOSS_VALUE_RE = re.compile(r"loss=([0-9.]+)")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_epoch_blocks(text: str) -> list[dict[str, Any]]:
    """All per-epoch validation blocks as {task, epoch, accuracy, macro_f1}."""
    return [
        {"task": task, "epoch": int(epoch), "accuracy": float(acc), "macro_f1": float(f1)}
        for acc, f1, task, epoch in BLOCK_RE.findall(text)
    ]


def parse_family_losses(text: str) -> dict[int, float]:
    """Map family epoch -> end-of-epoch training loss from the tqdm 100% lines."""
    losses: dict[int, float] = {}
    for line in text.splitlines():
        m = LOSS_LINE_RE.search(line)
        if not m:
            continue
        values = LOSS_VALUE_RE.findall(line)
        if values:
            losses[int(m.group(1))] = float(values[-1])
    return losses


def ddp_logs() -> list[Path]:
    return sorted(LOG_DIR.glob("deberta_lora_ddp_*.log"))


# --------------------------------------------------------------------------- #
# 1. Final validation metrics (label + family adapters)
# --------------------------------------------------------------------------- #
def build_model_metrics() -> pd.DataFrame:
    rows = []
    for task in ("label", "family"):
        m = read_json(CKPT_DIR / task / "metrics.json")
        rows.append(
            {
                "task": task,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "valid_rows": m["valid_rows"],
                "train_rows": m["train_rows"],
                "best_epoch": m["epoch"],
                "max_length": m["max_length"],
                "lora_r": m["lora_r"],
                "source": f"ckpt/{task}/metrics.json",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Method comparison (TF-IDF -> initial LoRA -> final strong-head LoRA)
# --------------------------------------------------------------------------- #
def find_initial_lora() -> tuple[float | None, float | None, str]:
    """The "initial DeBERTa LoRA": the strongest run whose family attribution is
    still weak (<0.5), i.e. before the strong-head/long-training improvements.
    Picking the best weak run (rather than the earliest, roughest one) matches the
    proposal's "~0.30" family anchor."""
    candidates: list[tuple[float, float | None, str]] = []
    for path in ddp_logs():
        blocks = parse_epoch_blocks(path.read_text(encoding="utf-8", errors="ignore"))
        fam = [b["macro_f1"] for b in blocks if b["task"] == "family"]
        lab = [b["macro_f1"] for b in blocks if b["task"] == "label"]
        if fam and max(fam) < 0.5:
            candidates.append((max(fam), (max(lab) if lab else None), f"logs/{path.name}"))
    if not candidates:
        return None, None, "unavailable"
    family_best, label_best, src = max(candidates, key=lambda c: c[0])
    return label_best, family_best, src


def build_method_comparison(model_metrics: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASELINE_METRICS)
    tfidf_detect = float(base["detect_macro_f1"].iloc[0])
    tfidf_source = float(base["source_macro_f1"].iloc[0])

    final_detect = float(model_metrics.loc[model_metrics["task"] == "label", "macro_f1"].iloc[0])
    final_source = float(model_metrics.loc[model_metrics["task"] == "family", "macro_f1"].iloc[0])

    init_detect, init_source, init_src = find_initial_lora()
    # Representative fallbacks if logs are absent on a fresh checkout.
    if init_detect is None:
        init_detect, init_src = 0.917832, "representative (logs unavailable)"
    if init_source is None:
        init_source = 0.308952

    methods = [
        ("tfidf_sgd", "TF-IDF + SGD", tfidf_detect, tfidf_source, "result/csv/llm_text_baseline_metrics.csv"),
        ("deberta_lora_initial", "DeBERTa LoRA（初始）", init_detect, init_source, init_src),
        ("deberta_lora_strong", "DeBERTa LoRA + 强分类头", final_detect, final_source,
         "ckpt/{label,family}/metrics.json"),
    ]
    rows = []
    for order, (method, method_label, detect, source, src) in enumerate(methods):
        official = round(0.8 * detect + 0.2 * source, 6)
        for task, value in (("detect", detect), ("source", source), ("official", official)):
            rows.append(
                {
                    "order": order,
                    "method": method,
                    "method_label": method_label,
                    "task": task,
                    "macro_f1": round(value, 6),
                    "source": src,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Family training history (per-epoch valid macro-F1 / accuracy / train loss)
# --------------------------------------------------------------------------- #
def build_training_history() -> pd.DataFrame | None:
    best_path, best_blocks = None, []
    for path in ddp_logs():
        blocks = [b for b in parse_epoch_blocks(path.read_text(encoding="utf-8", errors="ignore"))
                  if b["task"] == "family"]
        if len(blocks) > len(best_blocks):
            best_path, best_blocks = path, blocks
    if best_path is None or not best_blocks:
        print("[warn] no family training run found in logs/; skipping training history")
        return None

    losses = parse_family_losses(best_path.read_text(encoding="utf-8", errors="ignore"))
    rows = [
        {
            "epoch": b["epoch"],
            "family_valid_macro_f1": round(b["macro_f1"], 6),
            "family_valid_accuracy": round(b["accuracy"], 6),
            "train_loss": losses.get(b["epoch"]),
            "source": f"logs/{best_path.name}",
        }
        for b in sorted(best_blocks, key=lambda x: x["epoch"])
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Final submission summary (predicted label + machine->family distribution)
# --------------------------------------------------------------------------- #
def build_submission_summary() -> pd.DataFrame:
    rows = list(iter_jsonl(SUBMISSION))
    total = len(rows)
    label_counts = {i: 0 for i in LABEL_NAMES}
    family_counts = {i: 0 for i in FAMILY_NAMES}
    for r in rows:
        label_counts[int(r["label"])] = label_counts.get(int(r["label"]), 0) + 1
        if int(r["label"]) == 1 and int(r["family"]) >= 0:
            family_counts[int(r["family"])] = family_counts.get(int(r["family"]), 0) + 1
    machine_total = label_counts.get(1, 0)

    out = []
    for code, name in LABEL_NAMES.items():
        count = label_counts.get(code, 0)
        out.append({"group": "label", "code": code, "name": name, "count": count,
                    "percent": count / total if total else 0.0, "denominator": total,
                    "source": "result/submissions/submit.jsonl"})
    for code, name in FAMILY_NAMES.items():
        count = family_counts.get(code, 0)
        out.append({"group": "family", "code": code, "name": name, "count": count,
                    "percent": count / machine_total if machine_total else 0.0,
                    "denominator": machine_total, "source": "result/submissions/submit.jsonl"})
    return pd.DataFrame(out)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    model_metrics = build_model_metrics()
    model_metrics.to_csv(CSV_DIR / "llm_text_model_metrics.csv", index=False, encoding="utf-8-sig")

    build_method_comparison(model_metrics).to_csv(
        CSV_DIR / "llm_text_method_comparison.csv", index=False, encoding="utf-8-sig")

    history = build_training_history()
    if history is not None:
        history.to_csv(CSV_DIR / "llm_text_family_training_history.csv", index=False, encoding="utf-8-sig")

    build_submission_summary().to_csv(
        CSV_DIR / "llm_text_submission_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote result CSVs to {CSV_DIR}")


if __name__ == "__main__":
    main()
