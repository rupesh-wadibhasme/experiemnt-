import os
import json
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# === bring FE (must exist and match training) ===
from FE_BN_embeding_beta import build_dataset_from_excel

# ====== config ======
OUT_DIR_DEFAULT       = "combo_ae_outputs"

MODEL_PATH_NAME       = "combo_autoencoder.keras"
META_JSON_NAME        = "meta.json"
COMBO_MAP_JSON_NAME   = "combo_map.json"
COMBO_STATS_JSON_NAME = "combo_stats.json"
THRESHOLDS_JSON_NAME  = "combo_threshold.json"

# Optional (only needed if you trained separate embeddings and saved these maps)
ACCOUNT_MAP_JSON_NAME = "account_map.json"
BU_MAP_JSON_NAME      = "bu_map.json"
CODE_MAP_JSON_NAME    = "code_map.json"

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

def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    Falls back to dataset-level median/iqr for unseen combos.
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    med_all, iqr_all = median_iqr(l)

    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 0.1)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)

def invert_pred_to_inr(df: pd.DataFrame, y_norm_pred: np.ndarray, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    amount_pred_inr = inv_signed_log1p( y_norm_pred * iqr_log_combo + median_log_combo )
    """
    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", 0.0) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", 1.0) for c in combo_list], dtype=np.float32)
    lv = y_norm_pred.reshape(-1) * iqr + med
    return inv_signed_log1p(lv)

def row_recon_error(X_true: np.ndarray, X_pred: np.ndarray, col_w: Optional[np.ndarray]) -> np.ndarray:
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
    """
    n_anom = len(anomaly_idx)
    n_cols = X_true.shape[1]

    feat1_name, feat1_act, feat1_rec = [None] * n_anom, [np.nan] * n_anom, [np.nan] * n_anom
    feat2_name, feat2_act, feat2_rec = [None] * n_anom, [np.nan] * n_anom, [np.nan] * n_anom

    err = (X_true - X_pred) ** 2
    if col_w is not None:
        err = err * col_w.reshape(1, -1)

    col_candidates = np.arange(n_cols)
    col_candidates = col_candidates[col_candidates != j_y]

    for i, ridx in enumerate(anomaly_idx):
        row_err = err[ridx, col_candidates]
        if not np.any(np.isfinite(row_err)):
            continue
        order = np.argsort(row_err)[::-1]
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
    bu   = row.get("BusinessUnitCode", "-")
    code = row.get("BankTransactionCode", "-")
    actual = row.get("amount", None)
    pred   = row.get("amount_pred", None)
    d_abs  = row.get("amount_diff_abs", None)
    d_pct  = row.get("amount_diff_pct", None)
    r      = row.get("recon_error", None)
    thr    = row.get("thr_recon", None)

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

# ====== count_norm (must match training_embedding_1_4.py) ======

def add_count_norm(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    count_day_col: str = "ValueDateKey",
    count_iqr_floor: float = 0.10,
    clip_lo: float = -10.0,
    clip_hi: float = 10.0,
) -> pd.DataFrame:
    """
    Adds:
      - trans_count_day: count per (combo_str, day)
      - count_norm: (log1p(count_day) - count_median_log_combo) / count_iqr_log_combo
    Uses TRAIN stats only, with _global fallback.
    """
    d = df.copy()

    if count_day_col in d.columns:
        day_key = d[count_day_col].astype(str)
    else:
        if "ts" not in d.columns:
            raise ValueError(f"Need {count_day_col} or ts column to compute daily counts.")
        day_key = pd.to_datetime(d["ts"]).dt.strftime("%Y%m%d")

    d["_day_key_tmp"] = day_key

    d["trans_count_day"] = (
        d.groupby(["combo_str", "_day_key_tmp"])["combo_str"]
         .transform("size")
         .astype("float32")
    )

    cnt_log = np.log1p(d["trans_count_day"].values.astype(np.float32))

    g = stats.get("_global", {})
    med_all = float(g.get("count_median_log", np.median(cnt_log) if len(cnt_log) else 0.0))
    iqr_all = float(g.get(
        "count_iqr_log",
        max((np.percentile(cnt_log, 75) - np.percentile(cnt_log, 25)) if len(cnt_log) else 1.0, count_iqr_floor)
    ))

    combo_list = d["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("count_median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("count_iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, count_iqr_floor)

    d["count_norm"] = ((cnt_log - med) / iqr).clip(clip_lo, clip_hi).astype("float32")

    d = d.drop(columns=["_day_key_tmp"])
    return d

# ====== mapping helpers ======

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def apply_combo_map(df: pd.DataFrame, combo_map: Dict[str, int]) -> pd.DataFrame:
    if OOV_ID_NAME not in combo_map:
        raise ValueError(f"combo_map is missing OOV key '{OOV_ID_NAME}'. Recreate artifacts from training.")
    oov_id = combo_map[OOV_ID_NAME]
    d = df.copy()
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype(int)
    return d

def _maybe_apply_cat_map(df: pd.DataFrame, col: str, map_obj: Optional[Dict[str, int]], out_col: str) -> pd.DataFrame:
    """
    If map_obj is provided, map df[col] (as str) to ids with OOV fallback.
    If map_obj is None, do not modify df (caller can fallback to zeros).
    """
    if map_obj is None:
        return df
    if OOV_ID_NAME not in map_obj:
        raise ValueError(f"{out_col} map missing OOV key '{OOV_ID_NAME}'.")
    oov_id = map_obj[OOV_ID_NAME]

    d = df.copy()
    if col not in d.columns:
        # if source col missing, make everything OOV
        d[out_col] = int(oov_id)
        return d

    s = d[col].astype(str).fillna("UNK")
    d[out_col] = s.map(map_obj).fillna(oov_id).astype("int32")
    return d

# ====== model input detection ======

def _model_expects_cat_ids(model) -> bool:
    """
    Detect whether the model's 2nd input is cat_ids (shape (None,4)) or combo_id (shape (None,) / scalar).
    """
    if len(model.inputs) < 2:
        raise ValueError("Loaded model has <2 inputs. Expected [x_in, combo_id/cat_ids].")

    shp = model.inputs[1].shape  # e.g. (None, 4) or (None,)
    # TensorFlow shape objects can be tricky; be defensive:
    try:
        return (len(shp) == 2) and (int(shp[1]) == 4)
    except Exception:
        # fallback: if rank is 2 and last dim is 4
        return (len(shp) == 2) and (shp[1] == 4)

# ====== col weights ======

def build_col_weights(
    tab_cols: List[str],
    j_y: int,
    weight_y_norm: float,
    weight_tabular: float,
    weight_count_norm: Optional[float] = None,
) -> np.ndarray:
    """
    Build per-column weights aligned to [tab_cols..., y_norm]
    Optionally weights count_norm specifically if present.
    """
    n_tab = len(tab_cols)
    col_w = np.full(n_tab + 1, float(weight_tabular), dtype=np.float32)
    col_w[j_y] = float(weight_y_norm)

    if weight_count_norm is not None and "count_norm" in tab_cols:
        j_cnt = tab_cols.index("count_norm")
        col_w[j_cnt] = float(weight_count_norm)

    return col_w

# ====== artifacts loading ======

def load_artifacts(model_dir: str):
    meta_path = os.path.join(model_dir, META_JSON_NAME)
    combo_map_path = os.path.join(model_dir, COMBO_MAP_JSON_NAME)
    stats_path = os.path.join(model_dir, COMBO_STATS_JSON_NAME)
    thr_path = os.path.join(model_dir, THRESHOLDS_JSON_NAME)

    for p, name in [(meta_path, "meta.json"), (combo_map_path, "combo_map.json"),
                    (stats_path, "combo_stats.json"), (thr_path, "combo_threshold.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found at {p}")

    meta = _load_json(meta_path)
    combo_map = _load_json(combo_map_path)
    stats_train = _load_json(stats_path)
    thr_obj = _load_json(thr_path)

    thr_global = float(thr_obj["thr_global"])
    thr_per_combo = thr_obj.get("thr_per_combo", {})

    tab_cols_train = meta.get("tabular_feature_cols", [])
    if not tab_cols_train:
        raise ValueError("meta.json missing 'tabular_feature_cols'")

    # hyperparams used in inference
    weight_y_norm   = float(meta.get("weight_y_norm", 5.0))
    weight_tabular  = float(meta.get("weight_tabular", 1.0))
    weight_count_norm = meta.get("weight_count_norm", None)  # only if you added this in training meta
    weight_count_norm = float(weight_count_norm) if weight_count_norm is not None else None

    abs_tol_inr     = float(meta.get("abs_tol_inr", 1.0))
    pct_tol         = float(meta.get("pct_tol", 0.05))
    mad_mult        = float(meta.get("mad_mult", 0.8))

    # count params (from meta if present)
    count_day_col   = str(meta.get("count_day_col", "ValueDateKey"))
    count_iqr_floor = float(meta.get("count_iqr_floor", 0.10))
    clip_lo         = float(meta.get("count_norm_clip_lo", -10.0))
    clip_hi         = float(meta.get("count_norm_clip_hi",  10.0))

    # model path from meta if present, else default
    model_rel = meta.get("model_path", MODEL_PATH_NAME)
    model_path = model_rel
    if not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, os.path.basename(model_rel))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path, compile=False)

    # Optional categorical maps
    account_map_path = os.path.join(model_dir, ACCOUNT_MAP_JSON_NAME)
    bu_map_path      = os.path.join(model_dir, BU_MAP_JSON_NAME)
    code_map_path    = os.path.join(model_dir, CODE_MAP_JSON_NAME)

    account_map = _load_json(account_map_path) if os.path.exists(account_map_path) else None
    bu_map      = _load_json(bu_map_path)      if os.path.exists(bu_map_path)      else None
    code_map    = _load_json(code_map_path)    if os.path.exists(code_map_path)    else None

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
        "weight_count_norm": weight_count_norm,
        "abs_tol_inr": abs_tol_inr,
        "pct_tol": pct_tol,
        "mad_mult": mad_mult,
        "count_day_col": count_day_col,
        "count_iqr_floor": count_iqr_floor,
        "count_norm_clip_lo": clip_lo,
        "count_norm_clip_hi": clip_hi,
        "account_map": account_map,
        "bu_map": bu_map,
        "code_map": code_map,
    }

# ====== inputs builder (AUTO supports old combo_id or new cat_ids) ======

def build_inputs_with_ynorm(
    df: pd.DataFrame,
    tab_cols: List[str],
    y_norm: np.ndarray,
    expects_cat_ids: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Returns:
      X_in: float32 matrix of [tabular_features, y_norm]
      X_aux:
        - if expects_cat_ids: int32 matrix (N,4) [combo_id, account_id, bu_id, code_id]
        - else: int32 vector (N,) combo_id
      col_names: names corresponding to X_in columns
      j_y: column index of y_norm within X_in
    """
    tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros((len(df), 0), dtype="float32")
    y = y_norm.reshape(-1, 1).astype("float32")
    X_in = np.hstack([tab, y]).astype("float32")

    col_names = list(tab_cols) + ["y_norm"]
    j_y = len(col_names) - 1

    if not expects_cat_ids:
        X_aux = df["combo_id"].astype("int32").values
        return X_in, X_aux, col_names, j_y

    # expects cat_ids (N,4)
    combo = df["combo_id"].astype("int32").values
    acct  = df["account_id"].astype("int32").values if "account_id" in df.columns else np.zeros(len(df), dtype="int32")
    bu    = df["bu_id"].astype("int32").values      if "bu_id" in df.columns      else np.zeros(len(df), dtype="int32")
    code  = df["code_id"].astype("int32").values    if "code_id" in df.columns    else np.zeros(len(df), dtype="int32")

    X_aux = np.stack([combo, acct, bu, code], axis=1).astype("int32")
    return X_in, X_aux, col_names, j_y

# ====== main scoring function ======

def score_bank_statement_df(
    df_raw: pd.DataFrame,
    model_dir: str = OUT_DIR_DEFAULT,
    output_path: Optional[str] = None,
    anomaly_col: str = ANOMALY_COL_NAME,
    reason_col: str = REASON_COL_NAME,
) -> pd.DataFrame:
    """
    Returns df_raw + 2 columns: Anomaly (0/1), AnomalyReason.
    """

    artifacts = load_artifacts(model_dir)
    model = artifacts["model"]

    expects_cat_ids = _model_expects_cat_ids(model)

    combo_map     = artifacts["combo_map"]
    stats_train   = artifacts["stats_train"]
    thr_global    = artifacts["thr_global"]
    thr_per_combo = artifacts["thr_per_combo"]

    tab_cols_train = artifacts["tab_cols_train"]

    weight_y_norm  = artifacts["weight_y_norm"]
    weight_tabular = artifacts["weight_tabular"]
    weight_count_norm = artifacts["weight_count_norm"]

    abs_tol_inr = artifacts["abs_tol_inr"]
    pct_tol     = artifacts["pct_tol"]
    mad_mult    = artifacts["mad_mult"]

    count_day_col   = artifacts["count_day_col"]
    count_iqr_floor = artifacts["count_iqr_floor"]
    clip_lo         = artifacts["count_norm_clip_lo"]
    clip_hi         = artifacts["count_norm_clip_hi"]

    account_map = artifacts["account_map"]
    bu_map      = artifacts["bu_map"]
    code_map    = artifacts["code_map"]

    # ---- base output: preserve original order ----
    df_out = df_raw.copy().reset_index(drop=True)
    df_out[anomaly_col] = 0
    df_out[reason_col] = ""

    # add row id so we can map back even if FE sorts
    df_fe_in = df_raw.copy().reset_index(drop=True)
    df_fe_in["__row_id__"] = np.arange(len(df_fe_in), dtype=int)

    # 1) FE
    feats_all, _, _ = build_dataset_from_excel(df_fe_in)

    # 2) Apply combo map
    feats_all = apply_combo_map(feats_all, combo_map)

    # 3) If your new model uses separate embeddings, create account_id/bu_id/code_id if possible
    #    This requires either FE already created these OR you saved *_map.json files.
    #    If maps exist, we map from the raw categorical columns (if present).
    if expects_cat_ids:
        # Try to map from commonly used column names in your pipeline:
        # adjust these if your FE uses different names.
        feats_all = _maybe_apply_cat_map(feats_all, "BankAccountCode", account_map, "account_id")
        feats_all = _maybe_apply_cat_map(feats_all, "BusinessUnitCode", bu_map, "bu_id")
        feats_all = _maybe_apply_cat_map(feats_all, "BankTransactionCode", code_map, "code_id")

        # If maps were not present, fallback to zeros (will run but loses those signals)
        if account_map is None and "account_id" not in feats_all.columns:
            feats_all["account_id"] = 0
        if bu_map is None and "bu_id" not in feats_all.columns:
            feats_all["bu_id"] = 0
        if code_map is None and "code_id" not in feats_all.columns:
            feats_all["code_id"] = 0

    # 4) Add count_norm using TRAIN stats (critical for alignment if training used it)
    if "count_norm" in tab_cols_train:
        feats_all = add_count_norm(
            feats_all,
            stats_train,
            count_day_col=count_day_col,
            count_iqr_floor=count_iqr_floor,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )

    # 5) Align tabular columns to training's set (order must match exactly)
    n_rows = len(feats_all)
    n_tab = len(tab_cols_train)
    if n_tab == 0:
        raise ValueError("Training tabular_feature_cols is empty; nothing to feed AE.")

    # create missing columns as 0.0
    for col in tab_cols_train:
        if col not in feats_all.columns:
            feats_all[col] = 0.0

    # enforce exact order in a view
    feats_for_model = feats_all.copy()
    feats_for_model[tab_cols_train] = feats_for_model[tab_cols_train].astype("float32")

    # 6) y_norm using TRAIN stats
    y_norm = normalize_amount(feats_for_model, stats_train)

    # 7) Build AE inputs
    X_in, X_aux, col_names, j_y = build_inputs_with_ynorm(
        feats_for_model, tab_cols_train, y_norm, expects_cat_ids=expects_cat_ids
    )

    # Column weights consistent with training (+ optional count_norm weighting)
    col_w = build_col_weights(
        tab_cols=tab_cols_train,
        j_y=j_y,
        weight_y_norm=weight_y_norm,
        weight_tabular=weight_tabular,
        weight_count_norm=weight_count_norm,
    )

    # 8) Model prediction + recon error
    X_pred = model.predict([X_in, X_aux], batch_size=2048, verbose=0)
    err = row_recon_error(X_in, X_pred, col_w)

    combo_arr = feats_for_model["combo_str"].astype(str).values
    thr_vec = np.array([thr_per_combo.get(c, thr_global) for c in combo_arr], dtype=float)
    mask_recon = err >= thr_vec

    # 9) INR-based tolerance check
    y_norm_pred = X_pred[:, j_y]
    amt_pred = invert_pred_to_inr(feats_for_model, y_norm_pred, stats_train)
    amt_act  = feats_for_model["amount"].astype(float).values

    mad_vec = np.array([stats_train.get(c, {}).get("mad_inr", 1.0) for c in combo_arr], dtype=float)

    tol_abs = np.maximum.reduce([
        np.full_like(amt_act, abs_tol_inr, dtype=float),
        mad_mult * mad_vec,
        pct_tol * np.abs(amt_act),
    ])
    diff_abs = np.abs(amt_pred - amt_act)
    mask_inr = diff_abs >= tol_abs

    is_anom = mask_recon & mask_inr
    idx = np.where(is_anom)[0]

    if len(idx) > 0:
        out = feats_for_model.iloc[idx].copy()
        out["amount_pred"] = amt_pred[idx]
        out["amount_diff_abs"] = diff_abs[idx]
        out["amount_diff_pct"] = out["amount_diff_abs"] / (np.abs(out["amount"]) + 1e-9)
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

        # Map back using __row_id__ (safe even if FE sorted)
        if "__row_id__" not in out.columns:
            raise ValueError("FE output missing '__row_id__'. Ensure inference injects it before FE call.")

        row_ids = out["__row_id__"].astype(int).values
        df_out.loc[row_ids, anomaly_col] = 1
        df_out.loc[row_ids, reason_col] = out["reason"].values

    # Optionally save
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(output_path, index=False)

    return df_out
