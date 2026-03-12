"""Interactive Plotly dashboard for customer segmentation analytics.

This module intentionally produces a single self-contained HTML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _fmt_currency_full(value: float) -> str:
    return f"${value:,.0f}"


def _fmt_currency_compact(value: float) -> str:
    """Human-friendly currency formatting.

    Examples:
    - 2_500_000 -> $2.5M
    - 450_000 -> $450K
    - 3_200 -> $3,200
    """

    sign = "-" if value < 0 else ""
    v = abs(float(value))

    if v >= 1_000_000:
        return f"{sign}${v / 1_000_000:.1f}M"
    if v >= 1_000:
        # Use K without decimals for readability.
        return f"{sign}${v / 1_000:.0f}K"
    return f"{sign}${v:,.0f}"


def _apply_money_axis(fig: go.Figure, axis: str = "y") -> None:
    """Format an axis using compact ticks + full currency in hover."""

    update = {
        f"{axis}axis_tickprefix": "$",
        f"{axis}axis_tickformat": "~s",
    }
    fig.update_layout(**update)


def build_segmentation_dashboard(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    insights: Optional[List[str]] = None,
    output_path: str | Path = Path("output/segmentation_dashboard.html"),
) -> Path:
    """Build and save the interactive dashboard as HTML."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_revenue = float(transactions["Revenue"].sum())
    total_customers = int(customers["CustomerID"].nunique())
    avg_rev_per_customer = float(customers["TotalRevenue"].mean())
    avg_purchase_freq = float(customers["TotalTransactions"].mean())

    # Make figures look clean by default.
    px.defaults.template = "plotly_white"

    labels = {
        "Frequency": "Number of purchases",
        "Monetary": "Total customer revenue",
        "Recency": "Days since last purchase",
        "SegmentRevenue": "Revenue",
        "CustomerCount": "Customers",
        "Cluster": "Cluster",
    }

    # SECTION 2 — RFM DISTRIBUTIONS
    fig_recency = px.histogram(
        customers,
        x="Recency",
        nbins=50,
        title="Recency Distribution",
        labels=labels,
    )
    fig_frequency = px.histogram(
        customers,
        x="Frequency",
        nbins=50,
        title="Frequency Distribution",
        labels=labels,
    )
    fig_monetary = px.histogram(
        customers,
        x="Monetary",
        nbins=50,
        title="Monetary Distribution",
        labels=labels,
    )
    _apply_money_axis(fig_monetary, axis="x")

    # SECTION 3 — CUSTOMER SEGMENTATION
    seg_count = (
        customers.groupby("Segment", as_index=False)
        .agg(CustomerCount=("CustomerID", "nunique"), SegmentRevenue=("TotalRevenue", "sum"))
        .sort_values("SegmentRevenue", ascending=False)
    )

    fig_seg_rev = px.bar(
        seg_count,
        x="Segment",
        y="SegmentRevenue",
        title="Revenue by Customer Segment",
        labels=labels,
    )
    _apply_money_axis(fig_seg_rev, axis="y")
    fig_seg_rev.update_traces(
        hovertemplate="Segment=%{x}<br>Revenue=%{y:$,.0f}<extra></extra>"
    )

    fig_seg_count = px.bar(
        seg_count,
        x="Segment",
        y="CustomerCount",
        title="Customer Count by Segment",
        labels=labels,
    )
    fig_seg_count.update_traces(hovertemplate="Segment=%{x}<br>Customers=%{y:,}<extra></extra>")

    # SECTION 4 — CLUSTER VISUALIZATION (RFM scatter plots)
    customers_for_plot = customers.copy()
    customers_for_plot["Cluster"] = customers_for_plot["Cluster"].astype(str)

    fig_cluster_fm = px.scatter(
        customers_for_plot,
        x="Frequency",
        y="Monetary",
        color="Cluster",
        title="Customer Value Clusters — Frequency vs Monetary",
        labels=labels,
        hover_data={
            "CustomerID": True,
            "Segment": True,
            "TotalRevenue": ":$,.0f",
            "Cluster": True,
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    _apply_money_axis(fig_cluster_fm, axis="y")

    fig_cluster_rm = px.scatter(
        customers_for_plot,
        x="Recency",
        y="Monetary",
        color="Cluster",
        title="Customer Behavior Segments — Recency vs Monetary",
        labels=labels,
        hover_data={
            "CustomerID": True,
            "Segment": True,
            "TotalRevenue": ":$,.0f",
            "Cluster": True,
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    _apply_money_axis(fig_cluster_rm, axis="y")

    # SECTION 5 — CUSTOMER VALUE (Pareto + top customers)
    pareto = customers[["CustomerID", "TotalRevenue"]].sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
    pareto = pareto[pareto["TotalRevenue"] > 0].copy()
    pareto["Rank"] = pareto.index + 1
    pareto["CumRevenuePct"] = 100.0 * pareto["TotalRevenue"].cumsum() / float(pareto["TotalRevenue"].sum())

    pareto_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pareto_fig.add_trace(
        go.Bar(
            x=pareto["Rank"],
            y=pareto["TotalRevenue"],
            name="Customer Revenue",
            hovertemplate="Rank=%{x}<br>Revenue=%{y:$,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    pareto_fig.add_trace(
        go.Scatter(
            x=pareto["Rank"],
            y=pareto["CumRevenuePct"],
            name="Cumulative Revenue %",
            mode="lines",
            hovertemplate="Rank=%{x}<br>Cumulative=%{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    pareto_fig.update_layout(
        title="Pareto Analysis — Revenue Contribution by Top Customers",
        xaxis_title="Customers (sorted by revenue)",
        yaxis_title="Revenue",
        legend_orientation="h",
        legend_y=-0.15,
        margin=dict(t=60, r=20, b=60, l=60),
    )
    pareto_fig.update_yaxes(tickprefix="$", tickformat="~s", secondary_y=False)
    pareto_fig.update_yaxes(range=[0, 100], ticksuffix="%", secondary_y=True)
    pareto_fig.add_hline(y=80, line_dash="dot", line_color="gray", secondary_y=True)
    top_customers = customers.sort_values("TotalRevenue", ascending=False).head(10)
    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["CustomerID", "TotalRevenue", "Transactions", "Recency", "Segment", "Cluster"],
                    align="left",
                ),
                cells=dict(
                    values=[
                        top_customers["CustomerID"],
                        top_customers["TotalRevenue"].map(lambda x: f"${x:,.0f}"),
                        top_customers["TotalTransactions"],
                        top_customers["Recency"],
                        top_customers.get("Segment", ""),
                        top_customers.get("Cluster", ""),
                    ],
                    align="left",
                ),
            )
        ]
    )
    table.update_layout(title="Top 10 Customers by Revenue")

    # Build HTML (multiple Plotly figures embedded).
    insights = insights or []
    insights_html = "\n".join([f"<li>{i}</li>" for i in insights]) if insights else "<li>No insights generated.</li>"

    kpi_html = f"""
        <div class='kpi-grid'>
            <div class='kpi'><div class='kpi-label'>Total Revenue</div><div class='kpi-value'>{_fmt_currency_compact(total_revenue)}</div></div>
            <div class='kpi'><div class='kpi-label'>Total Customers</div><div class='kpi-value'>{total_customers:,}</div></div>
            <div class='kpi'><div class='kpi-label'>Avg Revenue per Customer</div><div class='kpi-value'>{_fmt_currency_compact(avg_rev_per_customer)}</div></div>
            <div class='kpi'><div class='kpi-label'>Avg Purchase Frequency</div><div class='kpi-value'>{avg_purchase_freq:.2f}</div></div>
        </div>
    """

    # Build HTML. Use inline plotly.js to avoid Chrome/network issues.
    css = """
    <style>
            body { font-family: Arial, sans-serif; margin: 0; background: #fafafa; }
            .container { max-width: 1200px; margin: 0 auto; padding: 22px; }
            .section { background: #ffffff; border: 1px solid #eaeaea; border-radius: 12px; padding: 16px 16px; margin: 14px 0; }
            .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
            @media (min-width: 900px) { .grid-2 { grid-template-columns: 1fr 1fr; } }
            h1 { margin: 0 0 4px 0; }
            .subtitle { margin: 0 0 12px 0; color: #555; }
            .kpi-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
            @media (min-width: 900px) { .kpi-grid { grid-template-columns: repeat(4, 1fr); } }
            .kpi { border: 1px solid #efefef; border-radius: 12px; padding: 12px; background: #fff; }
            .kpi-label { font-size: 12px; color: #666; margin-bottom: 6px; }
            .kpi-value { font-size: 22px; font-weight: 700; }
            ul { margin-top: 8px; }
    </style>
    """

    # Include plotly.js once, then embed the rest without it.
    parts = [
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Customer Segmentation Dashboard</title>",
        css,
        "</head><body><div class='container'>",
        "<h1>Customer Segmentation Analytics</h1>",
        "<p class='subtitle'>RFM analysis + K-Means clustering on the Online Retail II dataset.</p>",

        "<div class='section'>",
        "<h2>SECTION 1 — KPI Summary</h2>",
        kpi_html,
        "</div>",

        "<div class='section'>",
        "<h2>SECTION 2 — RFM Distributions</h2>",
        "<div class='grid grid-2'>",
        fig_recency.to_html(full_html=False, include_plotlyjs=True),
        fig_frequency.to_html(full_html=False, include_plotlyjs=False),
        fig_monetary.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "</div>",

        "<div class='section'>",
        "<h2>SECTION 3 — Customer Segmentation</h2>",
        "<div class='grid grid-2'>",
        fig_seg_rev.to_html(full_html=False, include_plotlyjs=False),
        fig_seg_count.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "</div>",

        "<div class='section'>",
        "<h2>SECTION 4 — Cluster Visualization</h2>",
        "<div class='grid grid-2'>",
        fig_cluster_fm.to_html(full_html=False, include_plotlyjs=False),
        fig_cluster_rm.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "</div>",

        "<div class='section'>",
        "<h2>SECTION 5 — Customer Value</h2>",
        pareto_fig.to_html(full_html=False, include_plotlyjs=False),
        table.to_html(full_html=False, include_plotlyjs=False),
        "</div>",

        "<div class='section'>",
        "<h2>SECTION 6 — Insights</h2>",
        "<h3>Key Business Insights</h3>",
        f"<ul>{insights_html}</ul>",
        "</div>",

        "</div></body></html>",
    ]

    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
