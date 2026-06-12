"""Regenerate the code-produced course-report figures in one command.

Runs the result-metrics collector, then the paper-style figure builders used for
the final report: Group B dataset analysis and Group D results. The final Group
A and Group C figures are AI-generated assets kept under Illustration/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.collect_result_metrics import main as collect_results
from visualization import plot_paper_dataset, plot_paper_results


def main() -> None:
    collect_results()
    plot_paper_dataset.main()
    plot_paper_results.main()
    print("Code-produced report figures regenerated in Illustration/.")


if __name__ == "__main__":
    main()
