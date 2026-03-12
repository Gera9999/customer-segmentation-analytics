"""RFM analysis and rule-based segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RFMConfig:
    score_bins: int = 5


def aggregate_customers(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data at CustomerID level."""

    required = {"CustomerID", "InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "Revenue"}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}")

    grouped = transactions.groupby("CustomerID", as_index=False).agg(
        TotalRevenue=("Revenue", "sum"),
        TotalTransactions=("InvoiceNo", "nunique"),
        TotalProductsPurchased=("Quantity", "sum"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        LastPurchaseDate=("InvoiceDate", "max"),
    )

    return grouped


def compute_rfm(customer_agg: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute Recency/Frequency/Monetary from the aggregated customer table."""

    required = {"CustomerID", "TotalRevenue", "TotalTransactions", "LastPurchaseDate"}
    missing = sorted(required - set(customer_agg.columns))
    if missing:
        raise ValueError(f"Customer aggregation missing required columns: {missing}")

    out = customer_agg.copy()
    out["LastPurchaseDate"] = pd.to_datetime(out["LastPurchaseDate"], errors="coerce")

    if snapshot_date is None:
        snapshot_date = out["LastPurchaseDate"].max() + pd.Timedelta(days=1)

    out["Recency"] = (snapshot_date - out["LastPurchaseDate"]).dt.days.astype(int)
    out["Frequency"] = out["TotalTransactions"].astype(int)
    out["Monetary"] = out["TotalRevenue"].astype(float)

    return out


def _qcut_safe(series: pd.Series, q: int, labels) -> pd.Series:
    """Quantile binning that falls back to rank-based bins when duplicates exist."""

    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        ranked = series.rank(method="first")
        return pd.qcut(ranked, q=q, labels=labels, duplicates="drop")


def score_rfm(rfm: pd.DataFrame, config: RFMConfig = RFMConfig()) -> pd.DataFrame:
    """Create normalized RFM scores (1..score_bins) and a combined RFM score string."""

    out = rfm.copy()

    bins = int(config.score_bins)
    labels = list(range(1, bins + 1))

    # Recency: lower is better -> highest score for lowest recency.
    r_score = _qcut_safe(out["Recency"], q=bins, labels=list(reversed(labels))).astype(int)
    f_score = _qcut_safe(out["Frequency"], q=bins, labels=labels).astype(int)
    m_score = _qcut_safe(out["Monetary"], q=bins, labels=labels).astype(int)

    out["R_Score"] = r_score
    out["F_Score"] = f_score
    out["M_Score"] = m_score
    out["RFM_Score"] = out[["R_Score", "F_Score", "M_Score"]].astype(str).agg("".join, axis=1)

    # Optional normalized values for clustering/plotting convenience.
    for col in ["Recency", "Frequency", "Monetary"]:
        s = out[col].astype(float)
        out[f"{col}_Norm"] = (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    return out


def assign_segments(rfm_scored: pd.DataFrame) -> pd.DataFrame:
    """Assign rule-based marketing segments.

    Segments included per spec:
    Champions, Loyal Customers, Potential Loyalists, New Customers,
    At Risk, Hibernating.
    """

    out = rfm_scored.copy()

    r = out["R_Score"]
    f = out["F_Score"]
    m = out["M_Score"]

    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4),
        (r >= 3) & (f >= 4),
        (r >= 4) & (f.between(2, 3)),
        (r == 5) & (f <= 2),
        (r <= 2) & (f >= 3),
        (r <= 2) & (f <= 2),
    ]
    choices = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "New Customers",
        "At Risk",
        "Hibernating",
    ]

    out["Segment"] = np.select(conditions, choices, default="Other")
    return out
