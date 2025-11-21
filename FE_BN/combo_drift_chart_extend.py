# combo_drift_charts.py
# Build Combo-Level Behavioral Drift Charts
# - Uses bank_features_simple.build_dataset_from_excel to create training features
# - Reads anomalies CSV (output of training_combo_autoencoder.py)
# - From a given starting row (sorted by recon_error desc), finds up to N examples
#   whose (Account, BU, Code) exist in training set
# - For each, plots:
#     1) Amount drift chart: train amount history + test actual + model prediction
#     2) y_norm drift chart: train y_norm + test y_norm + predicted y_norm
#     3) Combo sensitivity chart: signed_log1p(amount) with median/IQR bands
# - Saves PNGs in combo_drift_plots/ and prints their paths.

import os
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# bring the simple FE (must be in PYTHONPATH / same folder)
from bank_features_simple import build_dataset_from_excel

OUT_DIR_DEFAULT = "combo_drift_plots"


# =========================
# helpers
# =========================

def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sanitize_for_filename(s: str) -> str:
    """
    Make a safe-ish filename fragment from combo string.
    """
    s = str(s)
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", s)
    return s[:80]  # clamp length


def load_training_features(train_excel_path: str, sheet_name: int = 0) -> pd.DataFrame:
    """
    Run the simple FE on the training Excel to get a feature dataframe
    with at least: ts, amount, BankAccountCode, BusinessUnitCode,
    BankTransactionCode, combo_str.
    """
    feats, tab_cols, combo_map = build_dataset_from_excel(
        train_excel_path, sheet_name=sheet_name
    )
    return feats


# ---------- stats / transforms (same logic as training script) ----------

def signed_log1p(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def median_iqr(x: np.ndarray) -> Tuple[float, float]:
    q50 = float(np.median(x))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = max(q75 - q25, 1e-6)
    return q50, iqr


def build_combo_stats_train(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute per-combo stats from TRAIN ONLY:
      - median_log and iqr_log of signed_log1p(amount)
      - count
    """
    stats: Dict[str, Dict[str, float]] = {}
    for combo, g in df_train.groupby("combo_str", sort=False):
        a = g["amount"].astype(float).values
        l = signed_log1p(a)
        med_log, iqr_log = median_iqr(l)
        stats[combo] = {
            "median_log": med_log,
            "iqr_log": iqr_log,
            "count": int(len(a)),
        }
    return stats


def normalize_amount_for_df(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    Uses combo-specific stats; if missing, falls back to global stats of this df.
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    # fallback globals on this df
    med_all, iqr_all = median_iqr(l)

    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 1e-6)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)


# =========================
# pick examples
# =========================

def pick_top_examples_with_training_combo(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    start_row: int = 0,
    num_examples: int = 3,
) -> List[Tuple[int, pd.Series]]:
    """
    From the predictions dataframe:
      - sort by recon_error desc (if available)
      - starting from 'start_row', walk rows
      - for each row, check if its combo exists in training df
      - collect up to 'num_examples' rows (original index + row Series)
    Returns a list of (row_idx_in_sorted, row_series).
