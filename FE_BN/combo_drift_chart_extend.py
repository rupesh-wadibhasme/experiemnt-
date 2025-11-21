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
    """
    if "recon_error" in pred_df.columns:
        pred_sorted = pred_df.sort_values("recon_error", ascending=False).reset_index(drop=True)
    else:
        pred_sorted = pred_df.reset_index(drop=True)

    required_cols = ["BankAccountCode", "BusinessUnitCode", "BankTransactionCode"]
    for c in required_cols:
        if c not in pred_sorted.columns:
            raise KeyError(f"Prediction file missing required column '{c}'.")
        if c not in train_df.columns:
            raise KeyError(f"Training features missing required column '{c}'.")

    picked: List[Tuple[int, pd.Series]] = []
    n = len(pred_sorted)
    i = max(0, int(start_row))

    while i < n and len(picked) < num_examples:
        row = pred_sorted.iloc[i]
        acct = str(row["BankAccountCode"])
        bu = str(row["BusinessUnitCode"])
        code = str(row["BankTransactionCode"])

        mask_train = (
            (train_df["BankAccountCode"].astype(str) == acct)
            & (train_df["BusinessUnitCode"].astype(str) == bu)
            & (train_df["BankTransactionCode"].astype(str) == code)
        )
        if mask_train.any():
            picked.append((i, row))
        i += 1

    if len(picked) < num_examples:
        print(f"[INFO] Only found {len(picked)} examples whose combo exists in training set.")

    return picked


# =========================
# plotting functions
# =========================

def plot_single_combo_drift(
    train_df: pd.DataFrame,
    anomaly_row: pd.Series,
    row_idx: int,
    out_dir: str,
    show: bool = False,
) -> str:
    """
    Amount drift chart:
      - Filter training df to same (Account, BU, Code)
      - Plot ts vs amount for training
      - Overlay anomaly actual amount and model predicted amount (if present)
    """
    acct = str(anomaly_row["BankAccountCode"])
    bu = str(anomaly_row["BusinessUnitCode"])
    code = str(anomaly_row["BankTransactionCode"])

    mask_train = (
        (train_df["BankAccountCode"].astype(str) == acct)
        & (train_df["BusinessUnitCode"].astype(str) == bu)
        & (train_df["BankTransactionCode"].astype(str) == code)
    )
    combo_train = train_df.loc[mask_train].copy()

    if combo_train.empty:
        raise ValueError("No training rows for this combo; should have been filtered earlier.")

    combo_train = combo_train.sort_values("ts")
    ts_train = pd.to_datetime(combo_train["ts"])
    amt_train = combo_train["amount"].astype(float).values

    ts_anom = pd.to_datetime(anomaly_row["ts"])
    amt_anom = float(anomaly_row.get("amount", np.nan))

    has_pred = "amount_pred" in anomaly_row.index
    amt_pred = float(anomaly_row["amount_pred"]) if has_pred else None

    plt.figure(figsize=(9, 4))
    plt.scatter(ts_train, amt_train, alpha=0.6, s=20, label="train amount")

    # anomaly actual
    plt.scatter(ts_anom, amt_anom, marker="o", s=80, edgecolors="black", label="test actual")

    # anomaly predicted
    if has_pred:
        plt.scatter(ts_anom, amt_pred, marker="x", s=90, label="model prediction")

    title = f"Combo drift: Acct={acct}, BU={bu}, Code={code}"
    plt.title(title)
    plt.xlabel("Date (ts)")
    plt.ylabel("Amount")
    plt.grid(True, alpha=0.3)
    plt.legend()

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_drift_row{row_idx}_{safe_combo}_amount.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


def plot_single_ynorm_drift(
    train_df: pd.DataFrame,
    stats_train: Dict[str, Dict[str, float]],
    anomaly_row: pd.Series,
    row_idx: int,
    out_dir: str,
    show: bool = False,
) -> str:
    """
    y_norm drift chart:
      - y_norm for training rows of this combo (using TRAIN stats)
      - y_norm for anomaly actual (from amount)
      - y_norm for anomaly prediction (from amount_pred, if present)
    """
    acct = str(anomaly_row["BankAccountCode"])
    bu = str(anomaly_row["BusinessUnitCode"])
    code = str(anomaly_row["BankTransactionCode"])

    # combo_str: use column if present, otherwise reconstruct
    if "combo_str" in anomaly_row.index:
        combo_str = str(anomaly_row["combo_str"])
    else:
        combo_str = f"{acct}|{bu}|{code}"

    mask_train = train_df["combo_str"].astype(str) == combo_str
    df_train_combo = train_df.loc[mask_train].copy()
    if df_train_combo.empty:
        raise ValueError("No training rows for this combo in y_norm drift.")

    df_train_combo = df_train_combo.sort_values("ts")

    # compute y_norm for training
    y_norm_train = normalize_amount_for_df(df_train_combo, stats_train)

    # anomaly actual as small df
    ts_anom = pd.to_datetime(anomaly_row["ts"])
    amt_anom = float(anomaly_row.get("amount", np.nan))
    df_test = pd.DataFrame(
        {
            "amount": [amt_anom],
            "combo_str": [combo_str],
            "ts": [ts_anom],
        }
    )
    y_norm_test = normalize_amount_for_df(df_test, stats_train)[0]

    # predicted y_norm from amount_pred
    has_pred = "amount_pred" in anomaly_row.index
    y_norm_pred = None
    if has_pred:
        amt_pred = float(anomaly_row["amount_pred"])
        df_pred = pd.DataFrame(
            {
                "amount": [amt_pred],
                "combo_str": [combo_str],
                "ts": [ts_anom],
            }
        )
        y_norm_pred = float(normalize_amount_for_df(df_pred, stats_train)[0])

    # plotting
    plt.figure(figsize=(9, 4))
    plt.scatter(
        pd.to_datetime(df_train_combo["ts"]),
        y_norm_train,
        s=20,
        alpha=0.6,
        label="train y_norm",
    )

    plt.scatter(
        [ts_anom],
        [y_norm_test],
        s=80,
        color="red",
        edgecolors="black",
        label="test y_norm (actual)",
    )

    if y_norm_pred is not None:
        plt.scatter(
            [ts_anom],
            [y_norm_pred],
            s=90,
            marker="x",
            color="green",
            label="predicted y_norm",
        )

    # median and 1-sigma bands in y_norm space
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1, label="median (0)")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    plt.axhline(-1.0, color="gray", linestyle=":", linewidth=1)

    title = f"y_norm Drift: Acct={acct}, BU={bu}, Code={code}"
    plt.title(title)
    plt.xlabel("Date (ts)")
    plt.ylabel("y_norm (normalized signed_log1p(amount))")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_drift_row{row_idx}_{safe_combo}_ynorm.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


def plot_single_combo_sensitivity(
    train_df: pd.DataFrame,
    stats_train: Dict[str, Dict[str, float]],
    anomaly_row: pd.Series,
    row_idx: int,
    out_dir: str,
    show: bool = False,
) -> str:
    """
    Combo sensitivity chart in log space:
      - signed_log1p(amount) for training rows of this combo
      - signed_log1p(amount) for anomaly actual
      - median_log and +/- 1/2 IQR bands
    """
    acct = str(anomaly_row["BankAccountCode"])
    bu = str(anomaly_row["BusinessUnitCode"])
    code = str(anomaly_row["BankTransactionCode"])

    if "combo_str" in anomaly_row.index:
        combo_str = str(anomaly_row["combo_str"])
    else:
        combo_str = f"{acct}|{bu}|{code}"

    mask_train = train_df["combo_str"].astype(str) == combo_str
    df_train_combo = train_df.loc[mask_train].copy()
    if df_train_combo.empty:
        raise ValueError("No training rows for this combo in sensitivity chart.")

    df_train_combo = df_train_combo.sort_values("ts")

    a_train = df_train_combo["amount"].astype(float).values
    log_train = signed_log1p(a_train)

    amt_test = float(anomaly_row.get("amount", np.nan))
    log_test = signed_log1p(np.array([amt_test]))[0]
    ts_test = pd.to_datetime(anomaly_row["ts"])

    # stats for this combo
    combo_stats = stats_train.get(combo_str, None)
    if combo_stats is None:
        med_log, iqr_log = median_iqr(log_train)
    else:
        med_log = combo_stats["median_log"]
        iqr_log = combo_stats["iqr_log"]

    plt.figure(figsize=(9, 4))
    plt.scatter(
        pd.to_datetime(df_train_combo["ts"]),
        log_train,
        s=20,
        alpha=0.6,
        color="blue",
        label="train signed_log1p(amount)",
    )

    plt.scatter(
        [ts_test],
        [log_test],
        s=80,
        color="red",
        edgecolors="black",
        label="test (actual, log-scale)",
    )

    # median + IQR bands
    plt.axhline(med_log, color="black", linestyle="--", linewidth=1, label="median_log")
    plt.axhline(med_log + iqr_log, color="gray", linestyle=":", linewidth=1, label="+1 IQR")
    plt.axhline(med_log - iqr_log, color="gray", linestyle=":", linewidth=1)
    plt.axhline(med_log + 2 * iqr_log, color="silver", linestyle=":", linewidth=1, label="+2 IQR")
    plt.axhline(med_log - 2 * iqr_log, color="silver", linestyle=":", linewidth=1)

    title = f"Combo Sensitivity (log amount): Acct={acct}, BU={bu}, Code={code}"
    plt.title(title)
    plt.xlabel("Date (ts)")
    plt.ylabel("signed_log1p(amount)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_drift_row{row_idx}_{safe_combo}_sensitivity.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


# =========================
# main entry
# =========================

def plot_combo_drift_charts(
    train_excel_path: str,
    pred_csv_path: str,
    start_row: int = 0,
    num_examples: int = 3,
    train_sheet: int = 0,
    out_dir: str = OUT_DIR_DEFAULT,
    show: bool = False,
):
    """
    Main entry:
      - train_excel_path: path to training history Excel (year1_full.xlsx)
      - pred_csv_path: path to anomalies CSV from AE script
      - start_row: index in sorted anomalies (by recon_error desc) to begin from
      - num_examples: how many examples to plot (default 3)
      - train_sheet: sheet index in training Excel (default 0)
      - out_dir: folder to save PNGs (default 'combo_drift_plots')
      - show: if True, show plots (useful in Jupyter); CLI keeps default False
    """
    _mkdir(out_dir)

    print(f"[INFO] Loading training features from: {train_excel_path}")
    train_df = load_training_features(train_excel_path, sheet_name=train_sheet)

    print(f"[INFO] Computing combo stats on training set...")
    stats_train = build_combo_stats_train(train_df)

    print(f"[INFO] Loading predictions from: {pred_csv_path}")
    pred_df = pd.read_csv(pred_csv_path)

    picked = pick_top_examples_with_training_combo(
        pred_df=pred_df,
        train_df=train_df,
        start_row=start_row,
        num_examples=num_examples,
    )

    if not picked:
        print("[WARN] No matching combos between predictions and training set.")
        return []

    all_paths: List[str] = []
    for row_idx, row in picked:
        try:
            p_amount = plot_single_combo_drift(
                train_df=train_df,
                anomaly_row=row,
                row_idx=row_idx,
                out_dir=out_dir,
                show=show,
            )
            p_ynorm = plot_single_ynorm_drift(
                train_df=train_df,
                stats_train=stats_train,
                anomaly_row=row,
                row_idx=row_idx,
                out_dir=out_dir,
                show=show,
            )
            p_sens = plot_single_combo_sensitivity(
                train_df=train_df,
                stats_train=stats_train,
                anomaly_row=row,
                row_idx=row_idx,
                out_dir=out_dir,
                show=show,
            )
            all_paths.extend([p_amount, p_ynorm, p_sens])
        except Exception as e:
            print(f"[WARN] Failed to plot for row {row_idx}: {e}")

    return all_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build combo-level behavioral drift charts.")
    parser.add_argument("--train_excel", required=True, help="Path to training Excel (full year).")
    parser.add_argument("--pred_csv", required=True, help="Path to predictions/anomalies CSV.")
    parser.add_argument("--start_row", type=int, default=0, help="Start index in sorted anomalies (default 0).")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to plot (default 3).")
    parser.add_argument("--train_sheet", type=int, default=0, help="Sheet index for training Excel (default 0).")
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Output folder for PNGs.")
    args = parser.parse_args()

    plot_combo_drift_charts(
        train_excel_path=args.train_excel,
        pred_csv_path=args.pred_csv,
        start_row=args.start_row,
        num_examples=args.num_examples,
        train_sheet=args.train_sheet,
        out_dir=args.out_dir,
        show=False,
    )
