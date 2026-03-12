"""K-Means clustering for customer segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KMeansConfig:
    k_range: Tuple[int, int] = (2, 10)  # inclusive
    random_state: int = 42
    n_init: int = 10


def compute_elbow_inertia(
    rfm_scored: pd.DataFrame,
    config: KMeansConfig = KMeansConfig(),
) -> Dict[int, float]:
    """Compute KMeans inertia for a range of k (Elbow Method)."""

    features = rfm_scored[["Recency", "Frequency", "Monetary"]].astype(float).values
    scaler = StandardScaler()
    x = scaler.fit_transform(features)

    k_min, k_max = config.k_range
    inertias: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=config.random_state, n_init=config.n_init)
        model.fit(x)
        inertias[k] = float(model.inertia_)

    return inertias


def choose_k_from_elbow(inertias: Dict[int, float]) -> int:
    """Choose k using a simple curvature heuristic (no extra dependencies)."""

    ks = np.array(sorted(inertias.keys()), dtype=float)
    ys = np.array([inertias[int(k)] for k in ks], dtype=float)

    if len(ks) < 3:
        return int(ks[0])

    # Normalize for numerical stability.
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-9)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)

    # Approximate second derivative magnitude.
    d2 = np.abs(np.diff(ys_n, n=2))
    best_idx = int(np.argmax(d2)) + 1
    return int(ks[best_idx])


def fit_kmeans(
    rfm_scored: pd.DataFrame,
    k: int,
    config: KMeansConfig = KMeansConfig(),
) -> pd.DataFrame:
    """Fit KMeans on standardized RFM and attach a Cluster label."""

    out = rfm_scored.copy()

    features = out[["Recency", "Frequency", "Monetary"]].astype(float).values
    scaler = StandardScaler()
    x = scaler.fit_transform(features)

    model = KMeans(n_clusters=int(k), random_state=config.random_state, n_init=config.n_init)
    labels = model.fit_predict(x)

    out["Cluster"] = labels.astype(int)
    return out
