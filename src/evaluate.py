"""
src/evaluate.py
Evaluation metrics and benchmark runner across models and split protocols.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.models import train_torch_mlp, train_geoepinet


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates Spearman rank correlation, Pearson correlation, and NDCG@10%."""
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)

    # Top-10% NDCG calculation
    k = max(1, int(len(y_true) * 0.10))
    p_order = np.argsort(y_pred)[::-1][:k]
    i_order = np.argsort(y_true)[::-1][:k]

    rel = y_true - min(np.min(y_true), 0.0)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[p_order] - 1) / discounts)
    idcg = np.sum((2 ** rel[i_order] - 1) / discounts)
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    return {
        "Spearman": round(float(rho), 4),
        "Pearson": round(float(r), 4),
        "NDCG@10%": round(float(ndcg), 4)
    }


def run_benchmark(
    df: pd.DataFrame, splits: dict,
    X_esm2: np.ndarray, X_esm3: np.ndarray,
    coords: np.ndarray, protein_name: str,
    device
) -> pd.DataFrame:
    """Runs all models across available splits and records performance metrics."""
    records = []
    y = df["DMS_score"].values
    max_len = len(coords)

    for split_name, (idx_tr, idx_te) in splits.items():
        if len(idx_tr) == 0 or len(idx_te) == 0:
            continue

        print(f"\n--- [{protein_name}] Evaluating {split_name.upper()} split ({len(idx_tr)} train / {len(idx_te)} test) ---")
        y_tr, y_te = y[idx_tr], y[idx_te]

        # 1. Ridge Baseline
        ridge_esm2 = Ridge(alpha=1.0).fit(X_esm2[idx_tr], y_tr)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM-2 + Ridge", **compute_metrics(y_te, ridge_esm2.predict(X_esm2[idx_te]))})

        ridge_esm3 = Ridge(alpha=1.0).fit(X_esm3[idx_tr], y_tr)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM3 + Ridge", **compute_metrics(y_te, ridge_esm3.predict(X_esm3[idx_te]))})

        # 2. XGBoost Baseline
        xgb_esm2 = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1).fit(X_esm2[idx_tr], y_tr)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM-2 + XGBoost", **compute_metrics(y_te, xgb_esm2.predict(X_esm2[idx_te]))})

        xgb_esm3 = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1).fit(X_esm3[idx_tr], y_tr)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM3 + XGBoost", **compute_metrics(y_te, xgb_esm3.predict(X_esm3[idx_te]))})

        # 3. Deep MLP Baseline
        mlp_esm2 = train_torch_mlp(X_esm2[idx_tr], y_tr, X_esm2[idx_te], device)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM-2 + Deep MLP", **compute_metrics(y_te, mlp_esm2)})

        mlp_esm3 = train_torch_mlp(X_esm3[idx_tr], y_tr, X_esm3[idx_te], device)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM3 + Deep MLP", **compute_metrics(y_te, mlp_esm3)})

        # 4. GeoEpiNet (Advanced Geometric Model)
        geo_preds = train_geoepinet(df, X_esm3, y, idx_tr, idx_te, coords, max_len, device)
        records.append({"Protein": protein_name, "Split": split_name, "Model": "ESM3 + GeoEpiNet", **compute_metrics(y_te, geo_preds)})

    return pd.DataFrame(records)