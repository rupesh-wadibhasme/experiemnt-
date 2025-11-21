# combo_drift_charts.py
# Build Combo-Level Behavioral Drift Charts
# - Uses bank_features_simple.build_dataset_from_excel to create training features
# - Reads anomalies CSV (output of training_combo_autoencoder.py)
# - From a given starting row (sorted by recon_error desc), finds up to N examples
#   whose (Account, BU, Code) exist in training set
# - For each, plots:
#     x: ts (date)
#     y: amount
#     train points, test actual, model prediction
# - Saves PNGs in combo_drift_plots/ and prints their paths.

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# bring the simple FE (must be in PYTHONPATH / same folder)
from bank_features_simple import build_dataset_from_excel

OUT_DIR_DEFAULT = "combo_drift_plots"


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
    with at least: ts, amount, BankAccountCode, BusinessUnitCode, BankTransactionCode, combo_str.
    """
    feats, tab_cols, combo_map = build_dataset_from_excel(train_excel_path, sheet_name=sheet_name)
    # Just return feats; we don't need tab_cols or combo_map here.
    return feats


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
        # if no recon_error, just keep original order
        pred_sorted = pred_df.reset_index(drop=True)

    required_cols = ["BankAccountCode", "BusinessUnitCode", "BankTransactionCode"]
    for c in required_cols:
        if c not in pred_sorted.columns:
            raise KeyError(f"Prediction file missing required column '{c}'.")

    for c in required_cols:
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
            (train_df["BankAccountCode"].astype(str) == acct) &
            (train_df["BusinessUnitCode"].astype(str) == bu) &
            (train_df["BankTransactionCode"].astype(str) == code)
        )
        if mask_train.any():
            picked.append((i, row))
        i += 1

    if len(picked) < num_examples:
        print(f"[INFO] Only found {len(picked)} examples whose combo exists in training set.")

    return picked


def plot_single_combo_drift(
    train_df: pd.DataFrame,
    anomaly_row: pd.Series,
    row_idx: int,
    out_dir: str,
) -> str:
    """
    For a single anomaly row:
      - Filter training df to same (Account, BU, Code)
      - Plot ts vs amount for training
      - Overlay anomaly actual amount and model predicted amount (if present)
      - Save PNG and return its path.
    """
    acct = str(anomaly_row["BankAccountCode"])
    bu   = str(anomaly_row["BusinessUnitCode"])
    code = str(anomaly_row["BankTransactionCode"])

    mask_train = (
        (train_df["BankAccountCode"].astype(str) == acct) &
        (train_df["BusinessUnitCode"].astype(str) == bu) &
        (train_df["BankTransactionCode"].astype(str) == code)
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
    fname = f"combo_drift_row{row_idx}_{safe_combo}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


def plot_combo_drift_charts(
    train_excel_path: str,
    pred_csv_path: str,
    start_row: int = 0,
    num_examples: int = 3,
    train_sheet: int = 0,
    out_dir: str = OUT_DIR_DEFAULT,
):
    """
    Main entry:
      - train_excel_path: path to training history Excel (year1_full.xlsx)
      - pred_csv_path: path to anomalies CSV from AE script
      - start_row: index in sorted anomalies (by recon_error desc) to begin from
      - num_examples: how many examples to plot (default 3)
      - train_sheet: sheet index in training Excel (default 0)
      - out_dir: folder to save PNGs (default 'combo_drift_plots')
    """
    _mkdir(out_dir)

    print(f"[INFO] Loading training features from: {train_excel_path}")
    train_df = load_training_features(train_excel_path, sheet_name=train_sheet)

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
            path = plot_single_combo_drift(
                train_df=train_df,
                anomaly_row=row,
                row_idx=row_idx,
                out_dir=out_dir,
            )
            all_paths.append(path)
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
    )
