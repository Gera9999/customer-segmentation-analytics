"""Data cleaning for retail transaction data."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def _normalize_col(col: str) -> str:
    return "".join(ch for ch in str(col).strip().lower() if ch.isalnum())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common Online Retail II column name variants.

    Examples seen in the wild:
    - Invoice -> InvoiceNo
    - Price -> UnitPrice
    - Customer ID -> CustomerID
    """

    rename_map = {}
    for c in df.columns:
        n = _normalize_col(c)
        if n in {"invoiceno", "invoice"}:
            rename_map[c] = "InvoiceNo"
        elif n in {"stockcode"}:
            rename_map[c] = "StockCode"
        elif n in {"description"}:
            rename_map[c] = "Description"
        elif n in {"quantity"}:
            rename_map[c] = "Quantity"
        elif n in {"invoicedate"}:
            rename_map[c] = "InvoiceDate"
        elif n in {"unitprice", "price"}:
            rename_map[c] = "UnitPrice"
        elif n in {"customerid", "customer"}:
            rename_map[c] = "CustomerID"
        elif n in {"country"}:
            rename_map[c] = "Country"

    if not rename_map:
        return df

    return df.rename(columns=rename_map)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transactions per portfolio project spec.

    - removes rows with missing CustomerID
    - removes negative quantities (returns)
    - removes transactions with UnitPrice <= 0
    - converts InvoiceDate to datetime
    - creates Revenue = Quantity * UnitPrice
    """

    df = standardize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()

    # Ensure types we rely on.
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    cleaned = cleaned.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])

    # Remove returns and invalid price rows.
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]

    # CustomerID sometimes appears as float (e.g. 12345.0). Keep as string-like id.
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int).astype(str)

    cleaned["Revenue"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    return cleaned
