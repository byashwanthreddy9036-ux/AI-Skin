"""Post-inference stages shared by `experiment`, `evaluate` and `report`.

Filled in by the evaluation (Phase 8) and reporting (Phase 9) stages.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from skin_research.config import AppConfig


def evaluate_and_report(cfg: AppConfig, predictions: pd.DataFrame, run_info: dict[str, Any]) -> dict[str, Any]:
    """Placeholder until evaluation is implemented: records that no metrics were computed."""
    return {"note": "evaluation not implemented yet"}
