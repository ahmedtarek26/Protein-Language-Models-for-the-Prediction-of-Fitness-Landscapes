"""
src/evaluate.py
Benchmark execution across foundation models, splits, and regression heads.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.models import train_torch_mlp


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates Spearman's rho, Pearson's r, and NDCG@10%."""
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


def run_benchmark(df: pd.DataFrame, splits: dict, X_esm2: np.ndarray, X_esm3: np.ndarray, protein_name: str, device) -> pd.DataFrame:
    """
    Fits and evaluates Ridge, XGBoost, and Deep MLP across all available splits.
    """
    records = []
    y = df["DMS_score"].values

    for split_name, (idx_tr, idx_te) in splits.items():
        if len(idx_tr) == 0 or len(idx_te) == 0:
            continue

        print(f"\n--- [{protein_name}] Evaluating {split_name.upper()} split ({len(idx_tr)} train / {len(idx_te)} test) ---")
        y_tr, y_te = y[idx_tr], y[idx_te]

        # Models to evaluate
        configs = [
            ("ESM-2", "Ridge", Ridge(alpha=1.0), X_esm2),
            ("ESM-2", "XGBoost", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1), X_esm2),
            ("ESM3", "Ridge", Ridge(alpha=1.0), X_esm3),
            ("ESM3", "XGBoost", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1), X_esm3),
        ]

        for plm, head_name, model, X_mat in configs:
            model.fit(X_mat[idx_tr], y_tr)
            preds = model.predict(X_mat[idx_te])
            metrics = compute_metrics(y_te, preds)
            records.append({"Protein": protein_name, "Split": split_name, "Foundation": plm, "Head": f"{plm} + {head_name}", **metrics})

        # Deep MLP
        mlp_esm2 = train_torch_mlp(X_esm2[idx_tr], y_tr, X_esm2[idx_te], device)
        records.append({"Protein": protein_name, "Split": split_name, "Foundation": "ESM-2", "Head": "ESM-2 + Deep MLP", **compute_metrics(y_te, mlp_esm2)})

        mlp_esm3 = train_torch_mlp(X_esm3[idx_tr], y_tr, X_esm3[idx_te], device)
        records.append({"Protein": protein_name, "Split": split_name, "Foundation": "ESM3", "Head": "ESM3 + Deep MLP", **compute_metrics(y_te, mlp_esm3)})

    return pd.DataFrame(records)