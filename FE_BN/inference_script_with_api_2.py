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

def _get_date_series(df: pd.DataFrame, col: str) -> pd.Series:
    # Normalize date-like columns to date (no time) where possible
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s).dt.date
    try:
        return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        return s

def add_count_norm(
    feats_all: pd.DataFrame,
    stats_train: Dict[str, Any],
    *,
    day_col: str = "ValueDateKey",
    out_col: str = "count_norm",
    raw_count_col: str = "trans_count_day",
    iqr_floor: float = 0.1,
    clip_lo: Optional[float] = None,
    clip_hi: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add a *normalized* per-day transaction count feature to an already-feature-engineered dataframe.

    Definition (per row):
        count_today = number of transactions for the same (Account, BU, Code, day)
        count_norm  = (log1p(count_today) - count_median_log_combo) / max(count_iqr_log_combo, iqr_floor)

    Notes:
    - Counts are computed on `feats_all` (post-cleaning / post-dedupe), so array lengths always match.
    - Uses per-combo `count_median_log` / `count_iqr_log` from `combo_stats.json`.
    - Falls back to `_global` stats if the combo is unseen; if `_global` is missing, falls back to the
      distribution of today's counts.
    """
    if out_col in feats_all.columns:
        return feats_all

    d = feats_all.copy()

    # --- choose day key column ---
    day_key_col = None
    if day_col in d.columns:
        day_key_col = day_col
    elif "ValueDateKey" in d.columns:
        day_key_col = "ValueDateKey"
    elif "PostingDateKey" in d.columns:
        day_key_col = "PostingDateKey"
    elif "ts" in d.columns:
        day_key_col = "_day_key"
        d[day_key_col] = pd.to_datetime(d["ts"], errors="coerce").dt.strftime("%Y%m%d")
    else:
        day_key_col = None

    base_cols = [c for c in ["BankAccountCode", "BusinessUnitCode", "BankTransactionCode"] if c in d.columns]
    group_cols = list(base_cols)
    if day_key_col is not None:
        group_cols.append(day_key_col)

    if len(group_cols) >= 3:
        base_col = "BankTransactionId" if "BankTransactionId" in d.columns else group_cols[0]
        d[raw_count_col] = d.groupby(group_cols)[base_col].transform("size").astype("float32")
    else:
        # fallback: count by combo only
        if "combo_str" in d.columns:
            d[raw_count_col] = d.groupby("combo_str")["combo_str"].transform("size").astype("float32")
        else:
            d[raw_count_col] = 1.0

    cnt_log = np.log1p(d[raw_count_col].values.astype(np.float32))

    if "combo_str" not in d.columns:
        d[out_col] = 0.0
        return d

    # --- defaults / fallbacks ---
    g = stats_train.get("_global", {}) if isinstance(stats_train, dict) else {}
    med_all = float(g.get("count_median_log", np.median(cnt_log) if len(cnt_log) else 0.0))
    q25 = float(np.percentile(cnt_log, 25)) if len(cnt_log) else 0.0
    q75 = float(np.percentile(cnt_log, 75)) if len(cnt_log) else 1.0
    iqr_today = max(q75 - q25, 1e-6)
    iqr_all = float(g.get("count_iqr_log", iqr_today))

    combo_list = d["combo_str"].astype(str).values
    med = np.array([stats_train.get(c, {}).get("count_median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats_train.get(c, {}).get("count_iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, float(iqr_floor))

    v = (cnt_log - med) / iqr
    if clip_lo is not None:
        v = np.maximum(v, float(clip_lo))
    if clip_hi is not None:
        v = np.minimum(v, float(clip_hi))

    d[out_col] = v.astype("float32")
    return d


def add_normalized_count_feature(
    df_raw: pd.DataFrame,
    feats_all: pd.DataFrame,
    stats_train: Dict[str, Any],
    *,
    out_col: str = "count_norm",
    count_col_raw: str = "trans_count_day",
    iqr_floor: float = 0.1,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    Prefer calling `add_count_norm(...)` directly.
    """
    _ = df_raw  # counts are computed on feats_all to avoid misalignment when FE drops duplicates
    return add_count_norm(
        feats_all,
        stats_train,
        day_col="ValueDateKey",
        out_col=out_col,
        raw_count_col=count_col_raw,
        iqr_floor=iqr_floor,
    )

    # Determine grouping columns available in df_raw
    base_cols = [c for c in ["BankAccountCode", "BusinessUnitCode", "BankTransactionCode"] if c in df_raw.columns]

    # Date column: ValueDateKey preferred
    date_col = "ValueDateKey" if "ValueDateKey" in df_raw.columns else ("PostingDateKey" if "PostingDateKey" in df_raw.columns else None)

    df_tmp = df_raw.copy()
    if date_col is not None:
        df_tmp[date_col] = _get_date_series(df_tmp, date_col)
        group_cols = base_cols + [date_col]
    else:
        group_cols = base_cols

    if len(group_cols) >= 3:
        base_col = "BankTransactionId" if "BankTransactionId" in df_tmp.columns else df_tmp.columns[0]
        cnt = df_tmp.groupby(group_cols)[base_col].transform("size").astype("float32")
    else:
        # Fallback: count by combo_str if present
        if "combo_str" in feats_all.columns:
            cnt = feats_all.groupby("combo_str")["combo_str"].transform("size").astype("float32")
        else:
            cnt = pd.Series(np.ones(len(df_tmp), dtype=np.float32))

    feats_all[count_col_raw] = cnt.values.astype("float32")
    cnt_log = np.log1p(feats_all[count_col_raw].values.astype(np.float32))

    if "combo_str" not in feats_all.columns:
        feats_all[out_col] = 0.0
        return feats_all

    combo_list = feats_all["combo_str"].astype(str).values
    med = np.array([stats_train.get(c, {}).get("count_median_log", 0.0) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats_train.get(c, {}).get("count_iqr_log", 1.0) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, iqr_floor)

    feats_all[out_col] = ((cnt_log - med) / iqr).astype("float32")
    return feats_all

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
    thr_path_candidates = [
        os.path.join(model_dir, THRESHOLDS_JSON_NAME),
        os.path.join(model_dir, "combo_threshold.json"),
    ]
    thr_path = next((p for p in thr_path_candidates if os.path.exists(p)), None)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    if not os.path.exists(combo_map_path):
        raise FileNotFoundError(f"combo_map.json not found at {combo_map_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"combo_stats.json not found at {stats_path}")
    if thr_path is None:
        raise FileNotFoundError(f"Thresholds file not found. Tried: {thr_path_candidates}")

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
    df_work = df_raw.copy()
    # Preserve a stable row key to map anomalies back to the original payload order.
    if "BankTransactionId" not in df_work.columns:
        df_work["__row_id__"] = np.arange(len(df_work), dtype=np.int64)

    feats_all, tab_cols_all, _ = build_dataset_from_excel(df_work)

    # drop ignored features for reconstruction
    tab_cols_all = [c for c in tab_cols_all if c not in IGNORE_FOR_RECON]

    # 1b) If training expects normalized count, compute it now (uses combo_stats.json)
    if "count_norm" in tab_cols_train:
        meta = artifacts.get("meta", {}) or {}
        iqr_floor = float(meta.get("count_iqr_floor", 0.1))
        clip_lo = meta.get("count_norm_clip_lo", None)
        clip_hi = meta.get("count_norm_clip_hi", None)
        day_col = meta.get("count_day_col", "ValueDateKey")
        feats_all = add_count_norm(
            feats_all,
            stats_train,
            day_col=day_col,
            out_col="count_norm",
            raw_count_col="trans_count_day",
            iqr_floor=iqr_floor,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )


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
    # Base output is the original payload (same rows / same order)
    df_out = df_raw.copy()
    df_out[anomaly_col] = 0
    df_out[reason_col] = ""

    # Stable key used for mapping FE rows -> output rows
    if "BankTransactionId" in df_out.columns and "BankTransactionId" in feats_all.columns:
        row_key_col = "BankTransactionId"
    else:
        row_key_col = "__row_id__"
        if row_key_col not in df_out.columns:
            df_out[row_key_col] = np.arange(len(df_out), dtype=np.int64)

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
        # Map anomalies back to df_out using the stable key (not by positional index)
        keys = out[row_key_col].values
        reason_map = dict(zip(keys, out["reason"].values))
        hit = df_out[row_key_col].isin(keys)
        df_out.loc[hit, anomaly_col] = 1
        df_out.loc[hit, reason_col] = df_out.loc[hit, row_key_col].map(reason_map).astype(str)

    # Optionally save
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)

    # Drop synthetic helper key if it was created for mapping
    if "__row_id__" in df_out.columns and "BankTransactionId" not in df_out.columns:
        df_out = df_out.drop(columns=["__row_id__"])

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
