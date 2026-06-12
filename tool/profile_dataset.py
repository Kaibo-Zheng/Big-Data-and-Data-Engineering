"""Profile the CCKS LLM-generated text detection dataset.

The raw files are JSONL:
- data/train.jsonl: text, label, id, split, optional method/model/family
- data/test_a_release.jsonl: text, id, split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "train.jsonl"
DEFAULT_TEST = ROOT / "data" / "test_a_release.jsonl"
ANALYSIS_DIR = ROOT / "analysis"
CSV_DIR = ANALYSIS_DIR / "csv"
REPORT_PATH = ANALYSIS_DIR / "llm_text_data_profile.md"

LABEL_NAMES = {
    0: "human",
    1: "machine",
    2: "hybrid",
}

SENTENCE_RE = re.compile(r"[.!?。！？]+")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc


def text_stats(text: str) -> dict[str, Any]:
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    sentence_count = max(1, len([x for x in SENTENCE_RE.split(text) if x.strip()]))
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    uppercase = sum(ch.isascii() and ch.isupper() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_len": (sum(len(w) for w in words) / word_count) if word_count else 0.0,
        "avg_sentence_words": word_count / sentence_count if sentence_count else 0.0,
        "uppercase_ratio": uppercase / ascii_letters if ascii_letters else 0.0,
        "digit_ratio": digits / char_count if char_count else 0.0,
        "punctuation_ratio": punctuation / char_count if char_count else 0.0,
    }


def load_rows(path: Path, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in iter_jsonl(path):
        text = obj.get("text", "")
        label = obj.get("label")
        row = {
            "dataset": dataset,
            "id": obj.get("id"),
            "split": obj.get("split"),
            "label": label,
            "label_name": LABEL_NAMES.get(label, "unknown") if label is not None else "",
            "method": obj.get("method", ""),
            "model": obj.get("model", ""),
            "family": obj.get("family", ""),
            "text_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        row.update(text_stats(text))
        rows.append(row)
    return pd.DataFrame(rows)


def quantiles(series: pd.Series) -> dict[str, float]:
    qs = series.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {
        "min": float(qs.loc[0.0]),
        "p25": float(qs.loc[0.25]),
        "p50": float(qs.loc[0.5]),
        "p75": float(qs.loc[0.75]),
        "p90": float(qs.loc[0.9]),
        "p99": float(qs.loc[0.99]),
        "max": float(qs.loc[1.0]),
        "mean": float(series.mean()),
    }


def file_overview(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": df["dataset"].iloc[0],
        "file": path.name,
        "file_size_mb": path.stat().st_size / 1024 / 1024,
        "rows": len(df),
        "unique_ids": df["id"].nunique(),
        "duplicate_text_rows": int(df["text_md5"].duplicated().sum()),
    }
    for prefix, col in [("char", "char_count"), ("word", "word_count")]:
        for key, value in quantiles(df[col]).items():
            out[f"{prefix}_{key}"] = value
    return out


def write_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: Path, test_path: Path) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat([train_df, test_df], ignore_index=True)

    overview = pd.DataFrame([
        file_overview(train_path, train_df),
        file_overview(test_path, test_df),
    ])
    overview.to_csv(CSV_DIR / "llm_text_overview.csv", index=False, encoding="utf-8-sig")

    label_dist = (
        train_df.groupby(["label", "label_name"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("label")
    )
    label_dist["percent"] = label_dist["rows"] / label_dist["rows"].sum()
    label_dist.to_csv(CSV_DIR / "llm_text_label_distribution.csv", index=False, encoding="utf-8-sig")

    method_dist = (
        train_df.assign(method=train_df["method"].replace("", "<missing>"))
        .groupby(["label", "label_name", "method"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["label", "rows"], ascending=[True, False])
    )
    method_dist["percent_in_label"] = method_dist["rows"] / method_dist.groupby("label")["rows"].transform("sum")
    method_dist.to_csv(CSV_DIR / "llm_text_method_distribution.csv", index=False, encoding="utf-8-sig")

    model_dist = (
        train_df.assign(model=train_df["model"].replace("", "<missing>"))
        .groupby(["label", "label_name", "model"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["label", "rows"], ascending=[True, False])
    )
    model_dist["percent_in_label"] = model_dist["rows"] / model_dist.groupby("label")["rows"].transform("sum")
    model_dist.to_csv(CSV_DIR / "llm_text_model_distribution.csv", index=False, encoding="utf-8-sig")

    family_dist = (
        train_df.assign(family=train_df["family"].replace("", "<missing>").astype(str))
        .groupby(["label", "label_name", "family"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["label", "family"])
    )
    family_dist.to_csv(CSV_DIR / "llm_text_family_distribution.csv", index=False, encoding="utf-8-sig")

    length_by_label = (
        train_df.groupby(["label", "label_name"], dropna=False)
        .agg(
            rows=("id", "count"),
            char_mean=("char_count", "mean"),
            char_p50=("char_count", "median"),
            word_mean=("word_count", "mean"),
            word_p50=("word_count", "median"),
            avg_sentence_words=("avg_sentence_words", "mean"),
            uppercase_ratio=("uppercase_ratio", "mean"),
            digit_ratio=("digit_ratio", "mean"),
            punctuation_ratio=("punctuation_ratio", "mean"),
        )
        .reset_index()
        .sort_values("label")
    )
    length_by_label.to_csv(CSV_DIR / "llm_text_length_by_label.csv", index=False, encoding="utf-8-sig")

    bins = [0, 50, 100, 150, 200, 300, 500, 800, 1200, 2000]
    hist_rows = []
    for dataset, df in [("train", train_df), ("test_a", test_df)]:
        counts = pd.cut(df["word_count"], bins=bins, right=False, include_lowest=True).value_counts().sort_index()
        for interval, count in counts.items():
            hist_rows.append({
                "dataset": dataset,
                "word_bin": str(interval),
                "rows": int(count),
                "percent": int(count) / len(df),
            })
    pd.DataFrame(hist_rows).to_csv(CSV_DIR / "llm_text_word_count_hist.csv", index=False, encoding="utf-8-sig")

    all_df.drop(columns=["text_md5"]).to_csv(CSV_DIR / "llm_text_row_stats.csv", index=False, encoding="utf-8-sig")

    quality = []
    for name, df in [("train", train_df), ("test_a", test_df)]:
        quality.append({
            "dataset": name,
            "rows": len(df),
            "missing_text": int((df["char_count"] == 0).sum()),
            "duplicate_text_rows": int(df["text_md5"].duplicated().sum()),
            "unique_ids": int(df["id"].nunique()),
            "id_min": int(df["id"].min()),
            "id_max": int(df["id"].max()),
        })
    pd.DataFrame(quality).to_csv(CSV_DIR / "llm_text_quality_checks.csv", index=False, encoding="utf-8-sig")

    write_markdown(overview, label_dist, method_dist, model_dist)


def write_markdown(
    overview: pd.DataFrame,
    label_dist: pd.DataFrame,
    method_dist: pd.DataFrame,
    model_dist: pd.DataFrame,
) -> None:
    lines = [
        "# LLM Text Detection Data Profile",
        "",
        "## Files",
        overview.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Label Distribution",
        label_dist.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Hybrid Construction Methods",
        method_dist[method_dist["label_name"] == "hybrid"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Machine Text Source Models",
        model_dist[model_dist["label_name"] == "machine"].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "- label=0: human text",
        "- label=1: machine-generated text",
        "- label=2: hybrid or edited text",
        "- data/test_a_release.jsonl has no labels and should be used only for final prediction output.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()

    train_df = load_rows(args.train, "train")
    test_df = load_rows(args.test, "test_a")
    write_outputs(train_df, test_df, args.train, args.test)
    print(f"Wrote CSV files to {CSV_DIR}")
    print(f"Wrote profile report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
