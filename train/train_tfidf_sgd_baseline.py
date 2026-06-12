"""Train a TF-IDF baseline for LLM text detection and source attribution.

Outputs:
- result/csv/llm_text_baseline_metrics.csv
- result/csv/llm_text_baseline_classification_report.csv
- result/csv/llm_text_baseline_confusion_matrix.csv
- result/csv/llm_text_baseline_source_report.csv
- result/submissions/test_a_tfidf_sgd_predictions.csv
- result/submissions/submit.jsonl
- visualization/figures/LLM_F07_baseline_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from visualization.style import apply_style, save


DEFAULT_TRAIN = ROOT / "data" / "train.jsonl"
DEFAULT_TEST = ROOT / "data" / "test_a_release.jsonl"
CSV_DIR = ROOT / "result" / "csv"
FIG_DIR = ROOT / "visualization" / "figures"
SUBMISSION_DIR = ROOT / "result" / "submissions"

LABEL_NAMES = {
    0: "human",
    1: "machine",
    2: "hybrid",
}

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

LABEL_DISPLAY_NAMES = {
    "human": "人类撰写",
    "machine": "机器生成",
    "hybrid": "人机协作",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_pipeline(max_features: int, min_df: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    max_iter=40,
                    tol=1e-3,
                    random_state=2026,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def probability_frame(pipeline: Pipeline, texts: list[str], names: dict[int, str], prefix: str) -> pd.DataFrame:
    proba = pipeline.predict_proba(texts)
    classes = [int(c) for c in pipeline.named_steps["clf"].classes_]
    data = {}
    for class_id, class_name in names.items():
        if class_id in classes:
            data[f"prob_{prefix}_{class_name}"] = proba[:, classes.index(class_id)]
        else:
            data[f"prob_{prefix}_{class_name}"] = 0.0
    return pd.DataFrame(data)


def official_scores(y_true: list[int], y_pred: list[int], family_true: list[int], family_pred: list[int]) -> dict[str, float]:
    detect_macro_f1 = f1_score(y_true, y_pred, labels=sorted(LABEL_NAMES), average="macro", zero_division=0)
    machine_positions = [i for i, label in enumerate(y_true) if label == 1]
    source_true = [family_true[i] for i in machine_positions]
    source_pred = [family_pred[i] if y_pred[i] == 1 else -1 for i in machine_positions]
    source_macro_f1 = f1_score(
        source_true,
        source_pred,
        labels=sorted(FAMILY_NAMES),
        average="macro",
        zero_division=0,
    )
    return {
        "detect_macro_f1": detect_macro_f1,
        "source_macro_f1": source_macro_f1,
        "official_final_score": 0.8 * detect_macro_f1 + 0.2 * source_macro_f1,
    }


def write_confusion_figure(cm: pd.DataFrame) -> None:
    apply_style()
    row_totals = cm.sum(axis=1).replace(0, 1)
    row_pct = cm.div(row_totals, axis=0)
    annot = cm.astype(int).astype(str) + "\n" + row_pct.map(lambda x: f"{x:.1%}")

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    cmap = sns.light_palette("#4F6D8A", as_cmap=True)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        square=True,
        cbar_kws={"label": "样本数", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("验证集混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticklabels([LABEL_DISPLAY_NAMES.get(x, x) for x in cm.columns], rotation=0)
    ax.set_yticklabels([LABEL_DISPLAY_NAMES.get(x, x) for x in cm.index], rotation=0)
    save(fig, FIG_DIR / "LLM_F07_baseline_confusion_matrix.png", source_csv="llm_text_baseline_confusion_matrix.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--family-max-features", type=int, default=120_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sample", type=int, default=0, help="Optional train row sample for quick checks.")
    args = parser.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train)
    if args.sample and args.sample < len(train_rows):
        train_rows = train_rows[: args.sample]

    labels = [int(r["label"]) for r in train_rows]

    train_split, valid_split = train_test_split(
        train_rows,
        test_size=args.test_size,
        random_state=2026,
        stratify=labels,
    )

    x_train = [r["text"] for r in train_split]
    y_train = [int(r["label"]) for r in train_split]
    x_valid = [r["text"] for r in valid_split]
    y_valid = [int(r["label"]) for r in valid_split]
    valid_family_true = [int(r.get("family", -1)) for r in valid_split]

    family_train_split = [r for r in train_split if int(r["label"]) == 1]
    x_family_train = [r["text"] for r in family_train_split]
    y_family_train = [int(r["family"]) for r in family_train_split]

    label_pipeline = build_pipeline(max_features=args.max_features, min_df=args.min_df)
    label_pipeline.fit(x_train, y_train)

    family_pipeline = build_pipeline(max_features=args.family_max_features, min_df=args.min_df)
    family_pipeline.fit(x_family_train, y_family_train)

    pred = [int(x) for x in label_pipeline.predict(x_valid)]
    family_pred = [int(x) for x in family_pipeline.predict(x_valid)]
    official = official_scores(y_valid, pred, valid_family_true, family_pred)

    metrics = pd.DataFrame(
        [
            {
                "model": "tfidf_sgd_multitask",
                "train_rows": len(x_train),
                "valid_rows": len(x_valid),
                "family_train_rows": len(x_family_train),
                "max_features": args.max_features,
                "family_max_features": args.family_max_features,
                "min_df": args.min_df,
                "accuracy": accuracy_score(y_valid, pred),
                "macro_f1": f1_score(y_valid, pred, labels=sorted(LABEL_NAMES), average="macro", zero_division=0),
                "weighted_f1": f1_score(y_valid, pred, average="weighted", zero_division=0),
                **official,
            }
        ]
    )
    metrics.to_csv(CSV_DIR / "llm_text_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    report = classification_report(
        y_valid,
        pred,
        labels=sorted(LABEL_NAMES),
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index(names="class")
    report_df.to_csv(CSV_DIR / "llm_text_baseline_classification_report.csv", index=False, encoding="utf-8-sig")

    valid_machine_positions = [i for i, label in enumerate(y_valid) if label == 1]
    source_true = [valid_family_true[i] for i in valid_machine_positions]
    source_pred = [family_pred[i] if pred[i] == 1 else -1 for i in valid_machine_positions]
    source_report = classification_report(
        source_true,
        source_pred,
        labels=sorted(FAMILY_NAMES),
        target_names=[FAMILY_NAMES[i] for i in sorted(FAMILY_NAMES)],
        output_dict=True,
        zero_division=0,
    )
    source_report_df = pd.DataFrame(source_report).T.reset_index(names="class")
    source_report_df.to_csv(CSV_DIR / "llm_text_baseline_source_report.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_valid, pred, labels=sorted(LABEL_NAMES))
    cm_df = pd.DataFrame(
        cm,
        index=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        columns=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    )
    cm_df.to_csv(CSV_DIR / "llm_text_baseline_confusion_matrix.csv", encoding="utf-8-sig")
    write_confusion_figure(cm_df)

    full_texts = [r["text"] for r in train_rows]
    full_labels = [int(r["label"]) for r in train_rows]
    full_family_rows = [r for r in train_rows if int(r["label"]) == 1]
    full_family_texts = [r["text"] for r in full_family_rows]
    full_family_labels = [int(r["family"]) for r in full_family_rows]

    label_submission_pipeline = build_pipeline(max_features=args.max_features, min_df=args.min_df)
    label_submission_pipeline.fit(full_texts, full_labels)
    family_submission_pipeline = build_pipeline(max_features=args.family_max_features, min_df=args.min_df)
    family_submission_pipeline.fit(full_family_texts, full_family_labels)

    test_rows = read_jsonl(args.test)
    test_texts = [r["text"] for r in test_rows]
    test_pred = [int(x) for x in label_submission_pipeline.predict(test_texts)]
    test_family_raw = [int(x) for x in family_submission_pipeline.predict(test_texts)]
    test_family = [family if label == 1 else -1 for label, family in zip(test_pred, test_family_raw)]
    label_proba = probability_frame(label_submission_pipeline, test_texts, LABEL_NAMES, "label")
    family_proba = probability_frame(family_submission_pipeline, test_texts, FAMILY_NAMES, "family")
    submission = pd.concat(
        [
            pd.DataFrame(
                {
                    "id": [r["id"] for r in test_rows],
                    "label": test_pred,
                    "family": test_family,
                    "label_name": [LABEL_NAMES[int(x)] for x in test_pred],
                    "family_name": [FAMILY_NAMES.get(int(x), "none") for x in test_family],
                }
            ),
            label_proba,
            family_proba,
        ],
        axis=1,
    )
    submission.to_csv(SUBMISSION_DIR / "test_a_tfidf_sgd_predictions.csv", index=False, encoding="utf-8-sig")
    with (SUBMISSION_DIR / "submit.jsonl").open("w", encoding="utf-8") as f:
        for row in submission[["id", "label", "family"]].to_dict("records"):
            f.write(json.dumps({"id": row["id"], "label": int(row["label"]), "family": int(row["family"])}, ensure_ascii=False))
            f.write("\n")

    print(metrics.to_string(index=False))
    print(f"Wrote outputs to {CSV_DIR}, {FIG_DIR}, and {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
