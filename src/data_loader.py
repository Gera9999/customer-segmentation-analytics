"""Data loading utilities for the Online Retail II dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_transactions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw transaction data.

    The Kaggle/UCI Online Retail II dataset is commonly distributed with a non-UTF8
    encoding and may include mixed dtypes.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ISO-8859-1 is a common encoding for this dataset.
    try:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8", errors="replace")

    return df
