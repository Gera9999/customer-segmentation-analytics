"""Business metrics and auto-generated insights."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def compute_kpis(transactions: pd.DataFrame, customers: pd.DataFrame) -> Dict[str, float]:
    """Compute dashboard KPIs from cleaned transactions and customer table."""

    total_revenue = float(transactions["Revenue"].sum())
    total_customers = float(customers["CustomerID"].nunique())

    avg_revenue_per_customer = float(customers["TotalRevenue"].mean())
    avg_purchase_frequency = float(customers["TotalTransactions"].mean())

    return {
        "TotalRevenue": total_revenue,
        "TotalCustomers": total_customers,
        "AvgRevenuePerCustomer": avg_revenue_per_customer,
        "AvgPurchaseFrequency": avg_purchase_frequency,
    }


def top_customers_by_revenue(customers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["CustomerID", "TotalRevenue", "TotalTransactions", "Recency", "Segment", "Cluster"]
    available = [c for c in cols if c in customers.columns]
    return customers.sort_values("TotalRevenue", ascending=False)[available].head(n).reset_index(drop=True)


def revenue_by_segment(customers: pd.DataFrame) -> pd.DataFrame:
    if "Segment" not in customers.columns:
        raise ValueError("Expected 'Segment' column on customers")

    return (
        customers.groupby("Segment", as_index=False)
        .agg(CustomerCount=("CustomerID", "nunique"), SegmentRevenue=("TotalRevenue", "sum"))
        .sort_values("SegmentRevenue", ascending=False)
        .reset_index(drop=True)
    )


def generate_insights(customers: pd.DataFrame) -> List[str]:
    """Generate portfolio-friendly business insights strings."""

    insights: List[str] = []

    total_rev = float(customers["TotalRevenue"].sum())
    if total_rev <= 0:
        return ["Total revenue is non-positive after cleaning; check input data filters."]

    if "Segment" in customers.columns:
        seg = revenue_by_segment(customers)
        top_seg = seg.iloc[0]
        pct = 100.0 * float(top_seg["SegmentRevenue"]) / total_rev
        insights.append(f"{top_seg['Segment']} generate {pct:.1f}% of total revenue.")

    # Revenue share from top 10% customers.
    n_customers = customers["CustomerID"].nunique()
    top_n = max(1, int(np.ceil(n_customers * 0.10)))
    top_rev = float(customers.sort_values("TotalRevenue", ascending=False).head(top_n)["TotalRevenue"].sum())
    insights.append(f"The top 10% of customers generate {100.0 * top_rev / total_rev:.1f}% of revenue.")

    # Pareto-style insight: smallest % of customers that generate 80% of revenue.
    rev_sorted = customers[["CustomerID", "TotalRevenue"]].sort_values("TotalRevenue", ascending=False)
    rev_sorted = rev_sorted[rev_sorted["TotalRevenue"] > 0]
    if not rev_sorted.empty:
        cum_pct = rev_sorted["TotalRevenue"].cumsum() / float(rev_sorted["TotalRevenue"].sum())
        idx_80 = int(np.searchsorted(cum_pct.values, 0.80, side="left"))
        idx_80 = min(max(idx_80, 0), len(rev_sorted) - 1)
        customers_needed = idx_80 + 1
        pct_customers = 100.0 * customers_needed / float(n_customers)
        pct_revenue = 100.0 * float(cum_pct.iloc[idx_80])
        insights.append(
            f"Pareto: Top {_fmt_pct(pct_customers)} of customers generate {_fmt_pct(pct_revenue)} of revenue."
        )

    avg_freq = float(customers["TotalTransactions"].mean())
    insights.append(f"Average purchase frequency is {avg_freq:.2f} transactions per customer.")

    # At-risk proxy using segment.
    if "Segment" in customers.columns:
        at_risk = customers[customers["Segment"].isin(["At Risk", "Hibernating"])]["CustomerID"].nunique()
        pct_risk = 100.0 * float(at_risk) / float(n_customers)
        insights.append(f"{pct_risk:.1f}% of customers are at risk of churn (At Risk + Hibernating).")

    # Largest cluster group.
    if "Cluster" in customers.columns:
        cluster_counts = customers["Cluster"].value_counts(dropna=True)
        if not cluster_counts.empty:
            largest_cluster = cluster_counts.index[0]
            pct_cluster = 100.0 * float(cluster_counts.iloc[0]) / float(n_customers)
            insights.append(f"Largest cluster is {largest_cluster} with {pct_cluster:.1f}% of customers.")

    return insights
