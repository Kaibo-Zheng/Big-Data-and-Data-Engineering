"""Run DeBERTa LoRA inference on data/test_a_release.jsonl."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "train" / "train_deberta_lora_single.py"

if len(sys.argv) == 1 or sys.argv[1] != "predict":
    sys.argv = [str(TARGET), "predict", *sys.argv[1:]]
else:
    sys.argv[0] = str(TARGET)

runpy.run_path(str(TARGET), run_name="__main__")
