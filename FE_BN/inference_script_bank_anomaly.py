import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# === bring FE (must exist and match training) ===
from FE_BN_embeding_1_1 import build_dataset_from_excel

# ====== config ======
OUT_DIR_DEFAULT       = "combo_ae_outputs"

MODEL_PATH_NAME       = "combo_autoencoder.keras"
META_JSON_NAME        = "meta.json"
COMBO_MAP_JSON_NAME   = "combo_map.json"
COMBO_STATS_JSON_NAME = "combo_stats.json"
THRESHOLDS_JSON_NAME  = "combo_thresholds.json"

# column names to add in API response
ANOMALY_COL_NAME      = "Anomaly"
REASON_COL_NAME       = "AnomalyReason"

# OOV combo id tag (must match training)
OOV_ID_NAME           = "__OOV__"

# features we DO NOT want in AE reconstruction / recon error (must match training)
IGNORE_FOR_RECON      = {"posting_lag_days", "cashbook_flag_derived", "quarter"}

# ====== basic utilities (copied from training to keep logic identical) ======

def signed_log1p(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))

def inv_signed_log1p(lv: np.ndarray) -> np.ndarray:
    return np.sign(lv) * (np.expm1(np.abs(lv)))

def median_iqr(x: np.ndarray) -> Tuple[float, float]:
    q50 = float(np.median(x))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = max(q75 - q25, 1e-6)
    return q50, iqr

def mad_inr(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(mad, 1e-6))

def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    Falls back to dataset-level median/iqr for unseen combos.
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    # dataset-level fallback
    med_all, iqr_all = median_iqr(l)

    combo_list = df["combo_str"].astype(str).values
    med = np.array(
        [stats.get(c, {}).get("median_log", med_all) for c in combo_list],
        dtype=np.float32,
    )
    iqr = np.array(
        [stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list],
        dtype=np.float32,
    )
    iqr = np.maximum(iqr, 0.1)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)

def invert_pred_to_inr(df: pd.DataFrame, y_norm_pred: np.ndarray, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    amount_pred_inr = inv_signed_log1p( y_norm_pred * iqr_log_combo + median_log_combo )
    """
    combo_list = df["combo_str"].astype(str).values
    med = np.array(
        [stats.get(c, {}).get("median_log", 0.0) for c in combo_list],
        dtype=np.float32,
    )
    iqr = np.array(
        [stats.get(c, {}).get("iqr_log", 1.0) for c in combo_list],
        dtype=np.float32,
    )
    lv = y_norm_pred.reshape(-1) * iqr + med
    return inv_signed_log1p(lv)

def build_inputs_with_ynorm(
    df: pd.DataFrame,
    tab_cols: List[str],
    y_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Returns:
      X_in: float32 matrix of [tabular_features, y_norm]  (this is what AE reconstructs)
      X_combo: int32 vector of combo_id                  (goes through embedding)
      col_names: names corresponding to X_in columns
      j_y: column index of y_norm within X_in
    """
    tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros(
        (len(df), 0), dtype="float32"
    )
    y = y_norm.reshape(-1, 1).astype("float32")
    X_in = np.hstack([tab, y]).astype("float32")
    X_combo = df["combo_id"].astype("int32").values
    col_names = list(tab_cols) + ["y_norm"]
    j_y = len(col_names) - 1
    return X_in, X_combo, col_names, j_y

def row_recon_error(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    col_w: Optional[np.ndarray],
) -> np.ndarray:
    """
    Weighted MSE per row (matches training loss definition).
    """
    if col_w is None:
        se = (X_true - X_pred) ** 2
        return se.mean(axis=1)
    se = ((X_true - X_pred) ** 2) * col_w.reshape(1, -1)
    return se.mean(axis=1)

def compute_top_feature_deviations(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    col_names: List[str],
    j_y: int,
    anomaly_idx: np.ndarray,
    col_w: Optional[np.ndarray],
    top_k: int = 2,
) -> Dict[str, List[Any]]:
    """
    For each anomaly row:
      - compute weighted squared error per feature
      - ignore y_norm index j_y (amount)
      - return top-k highest error features (name, actual, recon)
    Returns dict of column->list for direct assignment to anomalies df.
    """
    n_anom = len(anomaly_idx)
    n_cols = X_true.shape[1]

    feat1_name, feat1_act, feat1_rec = [None] * n_anom, [np.nan] * n_anom, [np.nan] * n_anom
    feat2_name, feat2_act, feat2_rec = [None] * n_anom, [np.nan] * n_anom, [np.nan] * n_anom

    err = (X_true - X_pred) ** 2
    if col_w is not None:
        err = err * col_w.reshape(1, -1)

    col_candidates = np.arange(n_cols)
    col_candidates = col_candidates[col_candidates != j_y]  # exclude amount/y_norm

    for i, ridx in enumerate(anomaly_idx):
        row_err = err[ridx, col_candidates]
        if not np.any(np.isfinite(row_err)):
            continue
        order = np.argsort(row_err)[::-1]  # descending
        if len(order) >= 1:
            j1 = col_candidates[order[0]]
            feat1_name[i] = col_names[j1]
            feat1_act[i] = float(X_true[ridx, j1])
            feat1_rec[i] = float(X_pred[ridx, j1])
        if len(order) >= 2 and top_k >= 2:
            j2 = col_candidates[order[1]]
            feat2_name[i] = col_names[j2]
            feat2_act[i] = float(X_true[ridx, j2])
            feat2_rec[i] = float(X_pred[ridx, j2])

    return {
        "top_feat1_name": feat1_name,
        "top_feat1_actual": feat1_act,
        "top_feat1_recon": feat1_rec,
        "top_feat2_name": feat2_name,
        "top_feat2_actual": feat2_act,
        "top_feat2_recon": feat2_rec,
    }

def format_reason(row: pd.Series) -> str:
    acct = row.get("BankAccountCode", "-")
    bu = row.get("BusinessUnitCode", "-")
    code = row.get("BankTransactionCode", "-")
    actual = row.get("amount", None)
    pred = row.get("amount_pred", None)
    d_abs = row.get("amount_diff_abs", None)
    d_pct = row.get("amount_diff_pct", None)
    r = row.get("recon_error", None)
    thr = row.get("thr_recon", None)
    try:
        pct_txt = f"{float(d_pct) * 100:.1f}%" if d_pct is not None else "-"
    except Exception:
        pct_txt = "-"

    base = (
        f"Amount looks unusual for (Account='{acct}', BU='{bu}', Code='{code}'). "
        f"Actual={actual}, Pred={pred}, DiffAbs={d_abs}, DiffPct={pct_txt}. "
        f"ReconError={r}, ThrRecon={thr}."
    )

    extras = []
    f1 = row.get("top_feat1_name", None)
    if isinstance(f1, str) and f1 != "":
        a1 = row.get("top_feat1_actual", None)
        p1 = row.get("top_feat1_recon", None)
        extras.append(f"{f1}: actual={a1}, recon={p1}")
    f2 = row.get("top_feat2_name", None)
    if isinstance(f2, str) and f2 != "":
        a2 = row.get("top_feat2_actual", None)
        p2 = row.get("top_feat2_recon", None)
        extras.append(f"{f2}: actual={a2}, recon={p2}")

    if extras:
        base += " Other deviating features: " + "; ".join(extras) + "."

    return base

def apply_combo_map(df: pd.DataFrame, combo_map: Dict[str, int]) -> pd.DataFrame:
    oov_id = combo_map[OOV_ID_NAME]
    d = df.copy()
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype(int)
    return d

# ====== helpers for artifacts / weights ======

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def build_col_weights(
    n_tab: int,
    j_y: int,
    weight_y_norm: float,
    weight_tabular: float,
) -> np.ndarray:
    col_w = np.full(n_tab + 1, weight_tabular, dtype=np.float32)
    col_w[j_y] = weight_y_norm
    return col_w

def load_artifacts(model_dir: str):
    """
    Load model + metadata + combo map + stats + thresholds.
    Expects files under model_dir with names defined above.
    """
    meta_path = os.path.join(model_dir, META_JSON_NAME)
    combo_map_path = os.path.join(model_dir, COMBO_MAP_JSON_NAME)
    stats_path = os.path.join(model_dir, COMBO_STATS_JSON_NAME)
    thr_path = os.path.join(model_dir, THRESHOLDS_JSON_NAME)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    if not os.path.exists(combo_map_path):
        raise FileNotFoundError(f"combo_map.json not found at {combo_map_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"combo_stats.json not found at {stats_path}")
    if not os.path.exists(thr_path):
        raise FileNotFoundError(f"combo_thresholds.json not found at {thr_path}")

    meta = _load_json(meta_path)
    combo_map = _load_json(combo_map_path)
    stats_train = _load_json(stats_path)
    thr_obj = _load_json(thr_path)

    thr_global = float(thr_obj["thr_global"])
    thr_per_combo = thr_obj.get("thr_per_combo", {})

    # hyperparams used in inference (fall back to defaults if missing)
    weight_y_norm = float(meta.get("weight_y_norm", 5.0))
    weight_tabular = float(meta.get("weight_tabular", 1.0))
    abs_tol_inr = float(meta.get("abs_tol_inr", 1.0))
    pct_tol = float(meta.get("pct_tol", 0.05))
    mad_mult = float(meta.get("mad_mult", 0.8))

    tab_cols_train = meta.get("tabular_feature_cols", [])
    if not tab_cols_train:
        raise ValueError("meta.json missing 'tabular_feature_cols'")

    # model path from meta if present, else default
    model_rel = meta.get("model_path", MODEL_PATH_NAME)
    model_path = model_rel
    if not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, os.path.basename(model_rel))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # load model without compiling (we only need forward pass)
    model = load_model(model_path, compile=False)

    return {
        "model": model,
        "meta": meta,
        "combo_map": combo_map,
        "stats_train": stats_train,
        "thr_global": thr_global,
        "thr_per_combo": thr_per_combo,
        "tab_cols_train": tab_cols_train,
        "weight_y_norm": weight_y_norm,
        "weight_tabular": weight_tabular,
        "abs_tol_inr": abs_tol_inr,
        "pct_tol": pct_tol,
        "mad_mult": mad_mult,
    }

# ====== main scoring function ======

def score_bank_statement_df(
    df_raw: pd.DataFrame,
    model_dir: str = OUT_DIR_DEFAULT,
    output_path: Optional[str] = None,
    anomaly_col: str = ANOMALY_COL_NAME,
    reason_col: str = REASON_COL_NAME,
) -> pd.DataFrame:
    """
    Run AE-based anomaly detection on a single day's bank statement.

    Returns:
      DataFrame with the SAME columns as df_raw + two extra columns:
      - anomaly_col (0/1)
      - reason_col (string; non-empty only when anomaly_col == 1)
    """
    artifacts = load_artifacts(model_dir)
    model = artifacts["model"]
    combo_map = artifacts["combo_map"]
    stats_train = artifacts["stats_train"]
    thr_global = artifacts["thr_global"]
    thr_per_combo = artifacts["thr_per_combo"]
    tab_cols_train = artifacts["tab_cols_train"]
    weight_y_norm = artifacts["weight_y_norm"]
    weight_tabular = artifacts["weight_tabular"]
    abs_tol_inr = artifacts["abs_tol_inr"]
    pct_tol = artifacts["pct_tol"]
    mad_mult = artifacts["mad_mult"]

    # 1) FE: build features from raw df (must match training FE)
    feats_all, tab_cols_all, _ = build_dataset_from_excel(df_raw.copy())

    # drop ignored features for reconstruction
    tab_cols_all = [c for c in tab_cols_all if c not in IGNORE_FOR_RECON]

    # 2) Align tabular columns to training's set (order & size must match)
    n_rows = len(feats_all)
    n_tab = len(tab_cols_train)
    if n_tab == 0:
        raise ValueError("Training tabular_feature_cols is empty; nothing to feed AE.")

    X_tab = np.zeros((n_rows, n_tab), dtype=np.float32)
    for j, col in enumerate(tab_cols_train):
        if col in feats_all.columns:
            X_tab[:, j] = feats_all[col].astype("float32").values
        else:
            X_tab[:, j] = 0.0  # missing feature -> 0

    # build DataFrame only to reuse build_inputs_with_ynorm, but preserve aligned order
    df_feats_for_model = feats_all.copy()
    for j, col in enumerate(tab_cols_train):
        df_feats_for_model[col] = X_tab[:, j]

    # 3) Apply combo map to get combo_id
    df_feats_for_model = apply_combo_map(df_feats_for_model, combo_map)

    # 4) y_norm using TRAIN stats
    y_norm = normalize_amount(df_feats_for_model, stats_train)

    # 5) Build AE inputs
    X_in, X_combo, col_names, j_y = build_inputs_with_ynorm(
        df_feats_for_model, tab_cols_train, y_norm
    )

    # Column weights for recon error, consistent with training
    col_w = build_col_weights(
        n_tab=len(tab_cols_train),
        j_y=j_y,
        weight_y_norm=weight_y_norm,
        weight_tabular=weight_tabular,
    )

    # 6) Model prediction + recon error
    X_pred = model.predict([X_in, X_combo], batch_size=2048, verbose=0)
    err = row_recon_error(X_in, X_pred, col_w)

    combo_arr = df_feats_for_model["combo_str"].astype(str).values
    thr_vec = np.array(
        [thr_per_combo.get(c, thr_global) for c in combo_arr],
        dtype=float,
    )
    mask_recon = err >= thr_vec

    # 7) INR-based tolerance check
    y_norm_pred = X_pred[:, j_y]
    amt_pred = invert_pred_to_inr(df_feats_for_model, y_norm_pred, stats_train)
    amt_act = df_feats_for_model["amount"].astype(float).values

    mad_vec = np.array(
        [stats_train.get(c, {}).get("mad_inr", 1.0) for c in combo_arr],
        dtype=float,
    )

    tol_abs = np.maximum.reduce(
        [
            np.full_like(amt_act, abs_tol_inr, dtype=float),
            mad_mult * mad_vec,
            pct_tol * np.abs(amt_act),
        ]
    )
    diff_abs = np.abs(amt_pred - amt_act)
    mask_inr = diff_abs >= tol_abs

    is_anom = mask_recon & mask_inr
    idx = np.where(is_anom)[0]

    # --- KEY CHANGE: base output is ORIGINAL INPUT DF, not FE DF ---
    df_out = df_raw.copy()
    df_out[anomaly_col] = 0
    df_out[reason_col] = ""

    if len(idx) > 0:
        # build anomaly details (mirrors training script)
        out = df_feats_for_model.iloc[idx].copy()
        out["amount_pred"] = amt_pred[idx]
        out["amount_diff_abs"] = diff_abs[idx]
        out["amount_diff_pct"] = out["amount_diff_abs"] / (
            np.abs(out["amount"]) + 1e-9
        )
        out["recon_error"] = err[idx]
        out["thr_recon"] = thr_vec[idx]
        out["is_anomaly"] = 1

        dev = compute_top_feature_deviations(
            X_true=X_in,
            X_pred=X_pred,
            col_names=col_names,
            j_y=j_y,
            anomaly_idx=idx,
            col_w=col_w,
            top_k=2,
        )
        for k, vals in dev.items():
            out[k] = vals

        out["reason"] = out.apply(format_reason, axis=1)

        # propagate anomaly flag and reason back to *original* df using index alignment
        df_out.loc[out.index, anomaly_col] = 1
        df_out.loc[out.index, reason_col] = out["reason"].values

    # Optionally save
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)

    return df_out

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run combo AE inference on a bank statement file."
    )
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file (one day).")
    parser.add_argument("--output", required=True, help="Output CSV with anomaly flags.")
    parser.add_argument(
        "--model_dir",
        default=OUT_DIR_DEFAULT,
        help="Directory with model + artifacts (default: combo_ae_outputs).",
    )
    args = parser.parse_args()

    score_bank_statement_file(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
    )
