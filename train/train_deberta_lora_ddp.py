"""DDP DeBERTa LoRA training entry point.

Launch with torchrun. Prediction is intentionally kept in the single-process
entry point to avoid duplicate submission writes.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train._deberta_lora_core import run_cli


if __name__ == "__main__":
    run_cli(distributed=True, include_predict=False)
