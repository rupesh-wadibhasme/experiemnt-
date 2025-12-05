# combo_drift_charts.py
# Build Combo-Level Behavioral Drift Charts
# - Uses bank_features_simple.build_dataset_from_excel to create training features
# - Reads anomalies CSV (output of training_combo_autoencoder.py)
# - From a given starting row (sorted by recon_error desc), finds up to N examples
#   whose (Account, BU, Code) exist in training set
# - For each, plots:
#     1) Amount drift over time
#     2) y_norm drift over time (with ±1 IQR and ±2 IQR bands)
#     3) (optional) Combo sensitivity chart
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


# ========== small helpers ==========

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
    return feats


# ====== stats helpers (same logic as in training script) ======

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


def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    # global fallback
    med_all, iqr_all = median_iqr(l)

    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 1e-6)

    return ((l - med) / iqr).astype(np.float32)


def normalize_single_amount(amount: float, combo_str: str, stats: Dict[str, Dict[str, float]],
                            fallback_med: float, fallback_iqr: float) -> float:
    """
    Normalize a single amount for a given combo_str, using TRAIN stats.
    """
    l = float(signed_log1p(np.array([amount]))[0])
    st = stats.get(combo_str, {})
    med = st.get("median_log", fallback_med)
    iqr = max(st.get("iqr_log", fallback_iqr), 1e-6)
    return float((l - med) / iqr)


# ========== picking examples ==========

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


# ========== plotting helpers ==========

def plot_single_combo_drift(
    train_df: pd.DataFrame,
    combo_anoms_df: pd.DataFrame,
    row_idx: int,
    out_dir: str,
    show_inline: bool = False,
) -> str:
    """
    Amount vs time drift chart.

    Modified:
      - Instead of a single anomaly_row, takes all anomaly rows for this combo
        (combo_anoms_df) and plots all of them on the same chart.
    """
    if combo_anoms_df.empty:
        raise ValueError("combo_anoms_df is empty for this combo.")

    # Use first row to identify combo
    first = combo_anoms_df.iloc[0]
    acct = str(first["BankAccountCode"])
    bu   = str(first["BusinessUnitCode"])
    code = str(first["BankTransactionCode"])

    # Training slice for this combo
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

    # All anomalies for this combo
    combo_anoms_df = combo_anoms_df.copy()
    combo_anoms_df = combo_anoms_df.sort_values("ts")
    ts_anom_all = pd.to_datetime(combo_anoms_df["ts"])
    amt_anom_all = combo_anoms_df["amount"].astype(float).values

    has_pred = "amount_pred" in combo_anoms_df.columns
    if has_pred:
        amt_pred_all = combo_anoms_df["amount_pred"].astype(float).values
    else:
        amt_pred_all = None

    plt.figure(figsize=(9, 4))
    plt.scatter(ts_train, amt_train, alpha=0.6, s=20, label="train amount")
    # All anomaly actuals
    plt.scatter(ts_anom_all, amt_anom_all, marker="o", s=80, edgecolors="black", label="test actual")
    # All anomaly predictions (if present)
    if has_pred and amt_pred_all is not None:
        plt.scatter(ts_anom_all, amt_pred_all, marker="x", s=90, label="model prediction")

    title = f"Amount Drift: Acct={acct}, BU={bu}, Code={code}"
    plt.title(title)
    plt.xlabel("Date (ts)")
    plt.ylabel("Amount")
    plt.grid(True, alpha=0.3)
    plt.legend()

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_amount_drift_row{row_idx}_{safe_combo}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    if show_inline:
        plt.show()
    else:
        plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


def plot_single_y_norm_drift(
    train_df: pd.DataFrame,
    combo_anoms_df: pd.DataFrame,
    row_idx: int,
    stats_train: Dict[str, Dict[str, float]],
    out_dir: str,
    show_inline: bool = False,
) -> str:
    """
    y_norm drift chart with ±1 IQR and ±2 IQR bands.
    y_norm = normalized signed_log1p(amount) per combo.

    Modified:
      - Plots all anomaly rows for this combo (actual + predicted y_norm)
        on the same chart, instead of just a single anomaly.
    """
    if combo_anoms_df.empty:
        raise ValueError("combo_anoms_df is empty for this combo.")

    # Use first anomaly row to identify combo
    first = combo_anoms_df.iloc[0]
    acct = str(first["BankAccountCode"])
    bu   = str(first["BusinessUnitCode"])
    code = str(first["BankTransactionCode"])

    # Training slice for this combo
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

    # Fallback med/iqr from training slice
    a_train = combo_train["amount"].astype(float).values
    l_train = signed_log1p(a_train)
    fallback_med, fallback_iqr = median_iqr(l_train)

    # Train y_norm (unchanged logic)
    y_train = normalize_amount(combo_train, stats_train)

    # All anomaly points for this combo
    combo_anoms_df = combo_anoms_df.copy()
    combo_anoms_df = combo_anoms_df.sort_values("ts")
    ts_anom_all = pd.to_datetime(combo_anoms_df["ts"])

    if "combo_str" in combo_anoms_df.columns:
        combo_str = str(combo_anoms_df["combo_str"].iloc[0])
    else:
        combo_str = f"{acct}|{bu}|{code}"

    # Actual anomaly y_norms
    amt_actual_all = combo_anoms_df["amount"].astype(float).values
    y_actual_all = np.array(
        [
            normalize_single_amount(a, combo_str, stats_train, fallback_med, fallback_iqr)
            for a in amt_actual_all
        ],
        dtype=float,
    )

    # Predicted anomaly y_norms (if available)
    has_pred = "amount_pred" in combo_anoms_df.columns
    if has_pred:
        amt_pred_all = combo_anoms_df["amount_pred"].astype(float).values
        y_pred_all = np.array(
            [
                normalize_single_amount(a, combo_str, stats_train, fallback_med, fallback_iqr)
                for a in amt_pred_all
            ],
            dtype=float,
        )
    else:
        y_pred_all = None

    # Plot
    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    # IQR bands in normalized space: ±1 and ±2
    ax.axhspan(-1, 1, alpha=0.10, label="±1 IQR band")
    ax.axhspan(-2, 2, alpha=0.05, label="±2 IQR band")

    # median reference (0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    # train points
    ax.scatter(ts_train, y_train, alpha=0.6, s=20, label="train y_norm")

    # all test actuals
    ax.scatter(
        ts_anom_all,
        y_actual_all,
        marker="o",
        s=80,
        edgecolors="black",
        label="test y_norm (actual)",
    )

    # all predictions
    if has_pred and y_pred_all is not None:
        ax.scatter(
            ts_anom_all,
            y_pred_all,
            marker="x",
            s=90,
            label="predicted y_norm",
        )

    title = f"y_norm Drift: Acct={acct}, BU={bu}, Code={code}"
    ax.set_title(title)
    ax.set_xlabel("Date (ts)")
    ax.set_ylabel("y_norm (normalized signed_log1p(amount))")
    ax.grid(True, alpha=0.3)

    # nice y-limits to see bands and points
    values_for_min = [float(y_train.min()), float(y_actual_all.min()), -2.5]
    values_for_max = [float(y_train.max()), float(y_actual_all.max()), 2.5]

    if has_pred and y_pred_all is not None and len(y_pred_all) > 0:
        values_for_min.append(float(y_pred_all.min()))
        values_for_max.append(float(y_pred_all.max()))

    ymin = min(values_for_min)
    ymax = max(values_for_max)
    pad = 0.2 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)

    # handle duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_y_norm_drift_row{row_idx}_{safe_combo}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    if show_inline:
        plt.show()
    else:
        plt.close()

    print(f"[SAVED] {out_path}")
    return out_path


def plot_single_combo_sensitivity(
    train_df: pd.DataFrame,
    combo_anoms_df: pd.DataFrame,
    row_idx: int,
    out_dir: str,
    show_inline: bool = False,
) -> str:
    """
    Very simple 'sensitivity' view:
      - For the combo, show absolute amount over time (train + all anomalies).
    """
    if combo_anoms_df.empty:
        raise ValueError("combo_anoms_df is empty for this combo.")

    # Use first anomaly to identify combo
    first = combo_anoms_df.iloc[0]
    acct = str(first["BankAccountCode"])
    bu   = str(first["BusinessUnitCode"])
    code = str(first["BankTransactionCode"])

    # Training slice
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
    amt_abs_train = combo_train["amount"].abs().astype(float).values

    # All anomalies for this combo
    combo_anoms_df = combo_anoms_df.copy()
    combo_anoms_df = combo_anoms_df.sort_values("ts")
    ts_anom_all = pd.to_datetime(combo_anoms_df["ts"])
    amt_abs_anom_all = combo_anoms_df["amount"].abs().astype(float).values

    plt.figure(figsize=(9, 4))
    plt.scatter(ts_train, amt_abs_train, alpha=0.6, s=20, label="train |amount|")
    plt.scatter(
        ts_anom_all,
        amt_abs_anom_all,
        marker="o",
        s=80,
        edgecolors="black",
        label="test |amount|",
    )

    title = f"Combo Sensitivity: Acct={acct}, BU={bu}, Code={code}"
    plt.title(title)
    plt.xlabel("Date (ts)")
    plt.ylabel("|Amount|")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    safe_combo = _sanitize_for_filename(f"{acct}_{bu}_{code}")
    fname = f"combo_sensitivity_row{row_idx}_{safe_combo}.png"
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    if show_inline:
        plt.show()
    else:
        plt.close()

    print(f"[SAVED] {out_path}")
    return out_path

# ========== main entry ==========

def plot_combo_drift_charts(
    train_excel_path: str,
    pred_csv_path: str,
    start_row: int = 0,
    num_examples: int = 3,
    train_sheet: int = 0,
    out_dir: str = OUT_DIR_DEFAULT,
    show_y_norm: bool = True,
    show_sensitivity: bool = True,
    show_inline: bool = False,
):
    """
    Main entry:
      - train_excel_path: path to training history Excel (year1_full.xlsx)
      - pred_csv_path: path to anomalies CSV from AE script
      - start_row: index in sorted anomalies (by recon_error desc) to begin from
      - num_examples: how many combos to plot (based on top anomalies; <= num_examples)
      - train_sheet: sheet index in training Excel (default 0)
      - out_dir: folder to save PNGs (default 'combo_drift_plots')
      - show_y_norm: also plot y_norm drift with ±1/±2 IQR bands
      - show_sensitivity: also plot combo sensitivity chart
      - show_inline: if True, also display plots in notebook
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

    # Pre-compute TRAIN combo stats for y_norm
    stats_train = build_combo_stats_train(train_df)

    all_paths: List[str] = []

    # We now want exactly one set of plots per combo,
    # even if multiple picked rows belong to the same combo.
    plotted_combos = set()

    # If you have a toggle for amount drift elsewhere, keep using that.
    # Here we just assume it's always on.
    show_amount_drift = True

    for row_idx, row in picked:
        acct = str(row["BankAccountCode"])
        bu   = str(row["BusinessUnitCode"])
        code = str(row["BankTransactionCode"])
        combo_key = (acct, bu, code)

        if combo_key in plotted_combos:
            # Already plotted this combo (with all its anomalies)
            continue
        plotted_combos.add(combo_key)

        # All anomaly/prediction rows for this combo in the predictions file
        mask_pred_combo = (
            (pred_df["BankAccountCode"].astype(str) == acct) &
            (pred_df["BusinessUnitCode"].astype(str) == bu) &
            (pred_df["BankTransactionCode"].astype(str) == code)
        )
        combo_anoms_df = pred_df.loc[mask_pred_combo].copy()

        if combo_anoms_df.empty:
            print(f"[WARN] No prediction rows found for combo {combo_key}; skipping.")
            continue

        # Sort anomalies by time if ts present
        if "ts" in combo_anoms_df.columns:
            combo_anoms_df = combo_anoms_df.sort_values("ts")

        try:
            # Amount drift
            if show_amount_drift:
                p1 = plot_single_combo_drift(
                    train_df=train_df,
                    combo_anoms_df=combo_anoms_df,
                    row_idx=row_idx,
                    out_dir=out_dir,
                    show_inline=show_inline,
                )
                all_paths.append(p1)

            # y_norm drift
            if show_y_norm:
                p2 = plot_single_y_norm_drift(
                    train_df=train_df,
                    combo_anoms_df=combo_anoms_df,
                    row_idx=row_idx,
                    stats_train=stats_train,
                    out_dir=out_dir,
                    show_inline=show_inline,
                )
                all_paths.append(p2)

            # simple sensitivity view
            if show_sensitivity:
                p3 = plot_single_combo_sensitivity(
                    train_df=train_df,
                    combo_anoms_df=combo_anoms_df,
                    row_idx=row_idx,
                    out_dir=out_dir,
                    show_inline=show_inline,
                )
                all_paths.append(p3)

        except Exception as e:
            print(f"[WARN] Failed to plot for row {row_idx} (combo={combo_key}): {e}")

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
    parser.add_argument("--no_y_norm", action="store_true", help="Disable y_norm drift charts.")
    parser.add_argument("--no_sensitivity", action="store_true", help="Disable combo sensitivity charts.")
    args = parser.parse_args()

    plot_combo_drift_charts(
        train_excel_path=args.train_excel,
        pred_csv_path=args.pred_csv,
        start_row=args.start_row,
        num_examples=args.num_examples,
        train_sheet=args.train_sheet,
        out_dir=args.out_dir,
        show_y_norm=not args.no_y_norm,
        show_sensitivity=not args.no_sensitivity,
        show_inline=False,
    )
