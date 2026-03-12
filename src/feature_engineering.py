"""Feature engineering helpers.

Kept small on purpose: the project’s main engineered feature is Revenue.
"""

from __future__ import annotations

import pandas as pd


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add Revenue = Quantity * UnitPrice (idempotent)."""

    out = df.copy()
    if "Revenue" not in out.columns:
        out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    return out
