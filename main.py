"""Customer Segmentation Analytics Pipeline.

Runs end-to-end:
- Load + clean transactions
- Aggregate customer metrics
- RFM analysis + rule-based segments
- K-Means clustering on RFM
- Generate insights
- Build interactive Plotly dashboard

Usage:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

from dashboard.segmentation_dashboard import build_segmentation_dashboard
from src.clustering import KMeansConfig, choose_k_from_elbow, compute_elbow_inertia, fit_kmeans
from src.data_cleaning import clean_transactions
from src.data_loader import load_transactions
from src.insights import generate_insights
from src.rfm_analysis import RFMConfig, aggregate_customers, assign_segments, compute_rfm, score_rfm


def run(
    data_path: Path = Path("data/online_retail.csv"),
    dashboard_path: Path = Path("output/segmentation_dashboard.html"),
) -> Path:
    raw = load_transactions(data_path)
    transactions = clean_transactions(raw)

    customer_agg = aggregate_customers(transactions)
    rfm = compute_rfm(customer_agg)
    rfm_scored = score_rfm(rfm, config=RFMConfig(score_bins=5))
    segmented = assign_segments(rfm_scored)

    inertias = compute_elbow_inertia(segmented, config=KMeansConfig(k_range=(2, 10)))
    k = choose_k_from_elbow(inertias)

    customers = fit_kmeans(segmented, k=k)

    insights = generate_insights(customers)

    out_path = build_segmentation_dashboard(
        transactions=transactions,
        customers=customers,
        insights=insights,
        output_path=dashboard_path,
    )

    return out_path


if __name__ == "__main__":
    out = run()
    print(f"Dashboard saved to: {out}")
