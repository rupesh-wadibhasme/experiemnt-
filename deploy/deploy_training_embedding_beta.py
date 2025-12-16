import os, json, random
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, Model

# === bring FE (must exist) ===
from FE_BN_embeding_beta import build_dataset_from_excel

# ====== config ======
OUT_DIR                   = "combo_ae_outputs"
OUTPUT_CSV                = "anomalies_combo_ae_2024_109.csv"
LEARNING_CURVE_PNG        = "learning_curve.png"
META_JSON                 = "meta.json"
MODEL_PATH                = "combo_autoencoder.keras"
COMBO_MAP_JSON            = "combo_map.json"
COMBO_STATS_JSON          = "combo_stats.json"
THRESHOLDS_JSON            = "combo_threshold.json"

# split
VALID_FRAC_IN_TRAIN       = 0.10

# embedding
EMBED_DIM                 = 16

# model sizes
ENC_UNITS                 = (128, 64)
DEC_UNITS                 = (64, 128)
LR                        = 5e-4
BATCH_SIZE                = 512
EPOCHS                    = 200
PATIENCE                  = 10

# column weighting in loss
WEIGHT_YNORM              = 5.0   # emphasize the amount channel
WEIGHT_TABULAR            = 1.0

# recon threshold
THRESHOLD_PERCENTILE      = 99.0
MIN_SAMPLES_PER_COMBO_THR = 20

# INR adaptive tolerance
ABS_TOL_INR               = 1.0
PCT_TOL                   = 0.05
MAD_MULT                  = 0.8

# reproducibility
GLOBAL_SEED               = 42

# count normalization
COUNT_IQR_FLOOR      = 0.10
COUNT_NORM_CLIP_LO   = -10.0
COUNT_NORM_CLIP_HI   =  10.0
COUNT_DAY_COL        = "ValueDateKey"   # preferred; fallback to ts-derived day if missing

# OOV combo id tag
OOV_ID_NAME               = "__OOV__"

# features we DO NOT want in AE reconstruction / recon error
IGNORE_FOR_RECON          = {"posting_lag_days", "cashbook_flag_derived", "quarter"}
#IGNORE_FOR_RECON          = {"month_sin","month_cos","dow","dow_sin","dow_cos"}
# ====== utilities ======
def set_seed(s=GLOBAL_SEED):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def time_split_10m_train_2m_test(df: pd.DataFrame, ts_col="ts") -> Tuple[np.ndarray, np.ndarray]:
    d = df.sort_values(ts_col).copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    max_ts = d[ts_col].max()
    cutoff = max_ts - pd.DateOffset(months=2)
    test_mask = d[ts_col] >= cutoff
    train_mask = ~test_mask
    # align to original order
    return train_mask.values, test_mask.values

def train_valid_split_from_train_tail(df: pd.DataFrame,
                                      base_train_mask: np.ndarray,
                                      ts_col="ts",
                                      valid_frac=0.10) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.where(base_train_mask)[0]
    if len(idx) == 0:
        raise ValueError("Empty train slice.")
    sub = df.iloc[idx].sort_values(ts_col)
    n = len(sub)
    n_valid = max(1, int(round(n * valid_frac)))
    valid_idx = sub.index[-n_valid:]
    final_train = base_train_mask.copy()
    final_valid = np.zeros_like(base_train_mask, dtype=bool)
    final_valid[valid_idx] = True
    final_train[valid_idx] = False
    return final_train, final_valid

def split_train_valid_tail_full(df: pd.DataFrame, ts_col="ts", valid_frac=0.10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(ts_col).reset_index(drop=True)
    n = len(d)
    n_valid = max(1, int(round(n * valid_frac)))
    return d.iloc[: n - n_valid].reset_index(drop=True), d.iloc[n - n_valid :].reset_index(drop=True)

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

def build_combo_stats_train(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute per-combo stats from TRAIN ONLY:
      - median_log and iqr_log of signed_log1p(amount)
      - mad_inr on raw amount
      - count (rows per combo)
      - count_median_log and count_iqr_log of log1p(daily_count) across days per combo
      - _global fallback for count stats
    """
    stats: Dict[str, Dict[str, float]] = {}

    # ---- amount-based stats (existing) ----
    for combo, g in df_train.groupby("combo_str", sort=False):
        a = g["amount"].astype(float).values
        l = signed_log1p(a)
        med_log, iqr_log = median_iqr(l)
        mad = mad_inr(a)
        stats[combo] = {
            "median_log": med_log,
            "iqr_log": iqr_log,
            "mad_inr": mad,
            "count": int(len(a))
        }

    # ---- count stats across days (NEW) ----
    if COUNT_DAY_COL in df_train.columns:
        day_key = df_train[COUNT_DAY_COL]
    else:
        # fallback to day derived from ts
        if "ts" not in df_train.columns:
            raise ValueError("Need ValueDateKey or ts column to compute daily counts.")
        day_key = pd.to_datetime(df_train["ts"]).dt.strftime("%Y%m%d")

    daily = (
        pd.DataFrame({
            "combo_str": df_train["combo_str"].astype(str).values,
            "day_key": day_key.astype(str).values
        })
        .groupby(["combo_str", "day_key"])
        .size()
        .rename("count_day")
        .reset_index()
    )
    daily["count_log"] = np.log1p(daily["count_day"].astype("float32"))

    # global fallback
    g_med = float(np.median(daily["count_log"].values)) if len(daily) else 0.0
    g_iqr = float(np.percentile(daily["count_log"].values, 75) - np.percentile(daily["count_log"].values, 25)) if len(daily) else 1.0
    g_iqr = float(max(g_iqr, COUNT_IQR_FLOOR))
    stats["_global"] = {"count_median_log": g_med, "count_iqr_log": g_iqr}

    # per-combo daily distribution
    for combo, s in daily.groupby("combo_str")["count_log"]:
        med = float(np.median(s.values))
        iqr = float(np.percentile(s.values, 75) - np.percentile(s.values, 25))
        iqr = float(max(iqr, COUNT_IQR_FLOOR))
        stats.setdefault(combo, {})
        stats[combo]["count_median_log"] = med
        stats[combo]["count_iqr_log"] = iqr

    return stats


def stats_to_jsonable(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for combo, d in stats.items():
        out[str(combo)] = {
            "median_log": float(d.get("median_log", 0.0)),
            "iqr_log": float(d.get("iqr_log", 1.0)),
            "mad_inr": float(d.get("mad_inr", 1.0)),
            "count": int(d.get("count", 0)),
            # NEW (safe defaults)
            "count_median_log": float(d.get("count_median_log", 0.0)),
            "count_iqr_log": float(d.get("count_iqr_log", 1.0)),
        }
    return out



def thresholds_to_jsonable(thr_global: float,
                           thr_per_combo: Dict[str, float]) -> Dict[str, Any]:
    """
    Convert thresholds to JSON-safe structure:
      {
        "thr_global": float,
        "thr_per_combo": { combo_str: float, ... }
      }
    """
    return {
        "thr_global": float(thr_global),
        "thr_per_combo": {str(c): float(v) for c, v in thr_per_combo.items()},
    }


def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    # fallback globals
    med_all, iqr_all = median_iqr(l)

    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 0.1)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)

def add_count_norm(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Adds:
      - trans_count_day: count per (combo_str, day)
      - count_norm: (log1p(count_day) - count_median_log_combo) / count_iqr_log_combo
    Uses TRAIN stats only, with _global fallback.
    """
    d = df.copy()

    # pick day key
    if COUNT_DAY_COL in d.columns:
        day_key = d[COUNT_DAY_COL].astype(str)
    else:
        if "ts" not in d.columns:
            raise ValueError("Need ValueDateKey or ts column to compute daily counts.")
        day_key = pd.to_datetime(d["ts"]).dt.strftime("%Y%m%d")

    # per-row daily count within the provided df
    d["_day_key_tmp"] = day_key
    d["trans_count_day"] = (
        d.groupby(["combo_str", "_day_key_tmp"])["combo_str"]
         .transform("size")
         .astype("float32")
    )
    cnt_log = np.log1p(d["trans_count_day"].values.astype(np.float32))

    g = stats.get("_global", {})
    med_all = float(g.get("count_median_log", np.median(cnt_log) if len(cnt_log) else 0.0))
    iqr_all = float(g.get("count_iqr_log", max((np.percentile(cnt_log, 75) - np.percentile(cnt_log, 25)) if len(cnt_log) else 1.0, COUNT_IQR_FLOOR)))

    combo_list = d["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("count_median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("count_iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, COUNT_IQR_FLOOR)

    d["count_norm"] = ((cnt_log - med) / iqr).clip(COUNT_NORM_CLIP_LO, COUNT_NORM_CLIP_HI).astype("float32")

    d = d.drop(columns=["_day_key_tmp"])
    return d



def invert_pred_to_inr(df: pd.DataFrame, y_norm_pred: np.ndarray, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    amount_pred_inr = inv_signed_log1p( y_norm_pred * iqr_log_combo + median_log_combo )
    """
    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", 0.0) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", 1.0)  for c in combo_list], dtype=np.float32)
    lv = y_norm_pred.reshape(-1) * iqr + med
    return inv_signed_log1p(lv)

def build_inputs_with_ynorm(df: pd.DataFrame,
                            tab_cols: List[str],
                            y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    UPDATED:
      X_in: float32 matrix of [tabular_features, y_norm]
      cat_ids: int32 matrix (N,4) holding [combo_id, account_id, bu_id, code_id]
      col_names: names corresponding to X_in columns
      j_y: column index of y_norm within X_in

    Backward-compatible behavior:
      - If account_id/bu_id/code_id are missing, they will be filled with 0.
      - combo_id must exist (your pipeline already ensures it).
    """
    tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros((len(df), 0), dtype="float32")
    y = y_norm.reshape(-1, 1).astype("float32")
    X_in = np.hstack([tab, y]).astype("float32")

    # Build cat id matrix expected by the updated model:
    # [combo_id, account_id, bu_id, code_id]
    combo = df["combo_id"].astype("int32").values

    if "account_id" in df.columns:
        acct = df["account_id"].astype("int32").values
    else:
        acct = np.zeros(len(df), dtype="int32")

    if "bu_id" in df.columns:
        bu = df["bu_id"].astype("int32").values
    else:
        bu = np.zeros(len(df), dtype="int32")

    if "code_id" in df.columns:
        code = df["code_id"].astype("int32").values
    else:
        code = np.zeros(len(df), dtype="int32")

    cat_ids = np.stack([combo, acct, bu, code], axis=1).astype("int32")

    col_names = list(tab_cols) + ["y_norm"]
    j_y = len(col_names) - 1
    return X_in, cat_ids, col_names, j_y


def build_sample_weights_for_recon(df: pd.DataFrame,
                                   y_norm: np.ndarray,
                                   stats: Dict[str, Dict[str, float]],
                                   n_tab: int,
                                   j_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row weights (favor small INR, rare combos) and per-column weights (emphasize y_norm over tabular).
      row_w = 1 / (log1p(|amount|)+1) * 1 / sqrt(count_combo + 1)
      col_w: vector of length n_in = n_tab + 1 ; tabular=WEIGHT_TABULAR ; y_norm=WEIGHT_YNORM
    """
    amount = df["amount"].astype(float).values
    combo = df["combo_str"].astype(str).values
    w_amt = 1.0 / (np.log1p(np.abs(amount)) + 1.0)
    cnt = np.array([stats.get(c, {}).get("count", 1) for c in combo], dtype=np.float32)
    w_combo = 1.0 / np.sqrt(cnt + 1.0)
    row_w = (w_amt * w_combo).astype("float32")
    row_w = row_w / (row_w.mean() + 1e-9)

    col_w = np.full(n_tab + 1, WEIGHT_TABULAR, dtype=np.float32)
    col_w[j_y] = WEIGHT_YNORM
    return row_w, col_w

def make_weighted_mse(col_w: np.ndarray):
    """
    Weighted MSE over the reconstruction vector. col_w is broadcast to batch.
    """
    cw = tf.constant(col_w.reshape(1, -1), dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        err = y_pred - y_true
        se = tf.square(err) * cw
        return tf.reduce_mean(tf.reduce_mean(se, axis=1))
    return loss_fn

def make_autoencoder(
    n_in: int,
    n_combos: int,
    n_accounts: int,
    n_busunits: int,
    n_codes: int,
    embed_dim: int = 16,
    enc_units=(128, 64),
    dec_units=(64, 128),
    lr: float = 1e-3,
    col_w: Optional[np.ndarray] = None,
    # optional per-field embedding dims 
    account_embed_dim: Optional[int] = None,
    bu_embed_dim: Optional[int] = None,
    code_embed_dim: Optional[int] = None,
) -> Model:
    """
    UPDATED:
    AE that reconstructs X_in (tabular + y_norm), conditioned on multiple embeddings:
      - combo_id, account_id, bu_id, code_id

    Inputs:
      - x_in: shape (n_in,) float32
      - cat_ids: shape (4,) int32  -> [combo_id, account_id, bu_id, code_id]
    """
    # defaults for per-cat embedding dims
    account_embed_dim = int(account_embed_dim or max(2, embed_dim // 2))
    bu_embed_dim      = int(bu_embed_dim      or max(2, embed_dim // 2))
    code_embed_dim    = int(code_embed_dim    or max(2, embed_dim // 2))

    inp_vec = layers.Input(shape=(n_in,), name="x_in")
    inp_ids = layers.Input(shape=(4,), dtype="int32", name="cat_ids")  # [combo, acct, bu, code]

    # slice ids
    combo_id = layers.Lambda(lambda x: x[:, 0], name="combo_id")(inp_ids)
    acct_id  = layers.Lambda(lambda x: x[:, 1], name="account_id")(inp_ids)
    bu_id    = layers.Lambda(lambda x: x[:, 2], name="bu_id")(inp_ids)
    code_id  = layers.Lambda(lambda x: x[:, 3], name="code_id")(inp_ids)

    # embeddings
    emb_combo = layers.Embedding(input_dim=max(int(n_combos), 1), output_dim=int(embed_dim), name="combo_emb")(combo_id)
    emb_acct  = layers.Embedding(input_dim=max(int(n_accounts), 1), output_dim=int(account_embed_dim), name="acct_emb")(acct_id)
    emb_bu    = layers.Embedding(input_dim=max(int(n_busunits), 1), output_dim=int(bu_embed_dim), name="bu_emb")(bu_id)
    emb_code  = layers.Embedding(input_dim=max(int(n_codes), 1), output_dim=int(code_embed_dim), name="code_emb")(code_id)

    emb_combo = layers.Flatten()(emb_combo)
    emb_acct  = layers.Flatten()(emb_acct)
    emb_bu    = layers.Flatten()(emb_bu)
    emb_code  = layers.Flatten()(emb_code)

    # encoder
    x = layers.Concatenate()([inp_vec, emb_combo, emb_acct, emb_bu, emb_code])
    for h in enc_units:
        x = layers.Dense(h, activation="relu")(x)
    bottleneck = layers.Dense(enc_units[-1], activation="relu", name="bottleneck")(x)

    # decoder
    y = bottleneck
    for h in dec_units:
        y = layers.Dense(h, activation="relu")(y)
    out = layers.Dense(n_in, activation=None, name="x_recon")(y)

    m = Model(inputs=[inp_vec, inp_ids], outputs=out, name="combo_autoencoder")

    loss_fn = losses.MeanSquaredError() if col_w is None else make_weighted_mse(col_w)
    m.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return m

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
):
    """
    For each anomaly row:
      - compute weighted squared error per feature
      - ignore y_norm index j_y (amount)
      - return top-k highest error features (name, actual, recon)
    Returns dict of column->list for direct assignment to anomalies df.
    """
    n_anom = len(anomaly_idx)
    n_cols = X_true.shape[1]

    feat1_name, feat1_act, feat1_rec = [None]*n_anom, [np.nan]*n_anom, [np.nan]*n_anom
    feat2_name, feat2_act, feat2_rec = [None]*n_anom, [np.nan]*n_anom, [np.nan]*n_anom

    # per-feature weighted squared error
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

def plot_curve(hist, path: str):
    plt.figure(figsize=(7,4))
    plt.plot(hist.history.get("loss", []), label="train")
    if "val_loss" in hist.history: plt.plot(hist.history["val_loss"], label="valid")
    plt.title("AE reconstruction loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

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
        pct_txt = f"{float(d_pct)*100:.1f}%" if d_pct is not None else "-"
    except Exception:
        pct_txt = "-"

    base = (
        f"Amount looks unusual for (Account='{acct}', BU='{bu}', Code='{code}'). "
        f"Actual={actual}, Pred={pred}, DiffAbs={d_abs}, DiffPct={pct_txt}. "
        f"ReconError={r}, ThrRecon={thr}."
    )

    # optional extra deviations
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

# ----- combo id mapping helpers (train-only map, OOV for unseen) -----
def make_combo_map_from_train(df_train: pd.DataFrame) -> Dict[str, int]:
    uniq = df_train["combo_str"].astype(str).unique().tolist()
    combo_map = {c: i for i, c in enumerate(uniq)}
    combo_map[OOV_ID_NAME] = len(combo_map)  # reserve last as OOV
    return combo_map

def apply_combo_map(df: pd.DataFrame, combo_map: Dict[str, int]) -> pd.DataFrame:
    oov_id = combo_map[OOV_ID_NAME]
    d = df.copy()
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype(int)
    return d

# ====== internal split pipeline (single DF) ======
def run_pipeline_internal_split_df(df_all: pd.DataFrame,
                                   valid_frac: float = VALID_FRAC_IN_TRAIN) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build features from a single DataFrame, split by time (~10m/2m), train on train+valid, test on tail.
    Returns: (anomalies_df, meta_dict)
    """
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) FE
    feats_all, tab_cols, _ = build_dataset_from_excel(df_all)
    tab_cols = [c for c in tab_cols if c not in IGNORE_FOR_RECON]

    # 2) Split
    base_train_mask, test_mask = time_split_10m_train_2m_test(feats_all, ts_col="ts")
    train_mask, valid_mask = train_valid_split_from_train_tail(
        feats_all, base_train_mask, ts_col="ts", valid_frac=valid_frac
    )

    df_train = feats_all.loc[train_mask].reset_index(drop=True)
    df_valid = feats_all.loc[valid_mask].reset_index(drop=True)
    df_test  = feats_all.loc[test_mask].reset_index(drop=True)

    # combo map from TRAIN
    combo_map = make_combo_map_from_train(df_train)
    df_train = apply_combo_map(df_train, combo_map)
    df_valid = apply_combo_map(df_valid, combo_map)
    df_test  = apply_combo_map(df_test,  combo_map)

    # 3) Per-combo stats (TRAIN ONLY)
    stats_train = build_combo_stats_train(df_train)

    # NEW: add count_norm to each split using TRAIN stats only (MUST be before X building)
    df_train = add_count_norm(df_train, stats_train)
    df_valid = add_count_norm(df_valid, stats_train)
    df_test  = add_count_norm(df_test,  stats_train)

    # ensure count_norm is part of AE tabular features (MUST be before X building)
    if "count_norm" not in tab_cols:
        tab_cols = tab_cols + ["count_norm"]

    # Save per-combo stats for inference
    combo_stats_path = os.path.join(OUT_DIR, COMBO_STATS_JSON)
    with open(combo_stats_path, "w") as f:
        json.dump(stats_to_jsonable(stats_train), f, indent=2)

    # 4) y_norm
    y_train = normalize_amount(df_train, stats_train)
    y_valid = normalize_amount(df_valid, stats_train)
    y_test  = normalize_amount(df_test,  stats_train)

    # 5) AE inputs (NOW includes count_norm)
    Xtr, Ctr, col_names, j_y = build_inputs_with_ynorm(df_train, tab_cols, y_train)
    Xva, Cva, _,   _         = build_inputs_with_ynorm(df_valid, tab_cols, y_valid)
    Xte, Cte, _,   _         = build_inputs_with_ynorm(df_test,  tab_cols, y_test)

    # 6) Weights (must use final tab_cols length)
    w_row_tr, col_w = build_sample_weights_for_recon(df_train, y_train, stats_train, n_tab=len(tab_cols), j_y=j_y)
    w_row_va, _     = build_sample_weights_for_recon(df_valid, y_valid, stats_train, n_tab=len(tab_cols), j_y=j_y)

    n_combos    = len(combo_map)
    n_accounts  = int(df_train["account_id"].max()) + 1
    n_busunits  = int(df_train["bu_id"].max()) + 1
    n_codes     = int(df_train["code_id"].max()) + 1

    # 7) Model
    model = make_autoencoder(
    n_in=Xtr.shape[1],
    n_combos=n_combos,
    n_accounts=n_accounts,
    n_busunits=n_busunits,
    n_codes=n_codes,
    embed_dim=EMBED_DIM,
    enc_units=ENC_UNITS,
    dec_units=DEC_UNITS,
    lr=LR,
    col_w=col_w
    )

    hist = model.fit(
        [Xtr, Ctr], Xtr,
        validation_data=([Xva, Cva], Xva, w_row_va),
        sample_weight=w_row_tr,
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, PATIENCE//3), min_lr=1e-5, verbose=1)
        ],
        verbose=1
    )
    model.save(os.path.join(OUT_DIR, MODEL_PATH), include_optimizer=False)
    plot_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # 8) Threshold from VALID
    Xva_pred = model.predict([Xva, Cva], batch_size=2048, verbose=0)
    err_va = row_recon_error(Xva, Xva_pred, col_w)
    thr_global = float(np.percentile(err_va, THRESHOLD_PERCENTILE))

    thr_per_combo: Dict[str, float] = {}
    df_thr = pd.DataFrame({"combo": df_valid["combo_str"].astype(str).values, "err": err_va})
    for c, g in df_thr.groupby("combo"):
        if len(g) >= MIN_SAMPLES_PER_COMBO_THR:
            thr_per_combo[c] = float(np.percentile(g["err"].values, THRESHOLD_PERCENTILE))

    # Save thresholds for inference
    thr_obj = thresholds_to_jsonable(thr_global, thr_per_combo)
    thresholds_path = os.path.join(OUT_DIR, THRESHOLDS_JSON)
    with open(thresholds_path, "w") as f:
        json.dump(thr_obj, f, indent=2)

    # 9) Score TEST
    Xte_pred = model.predict([Xte, Cte], batch_size=2048, verbose=0)
    err_te = row_recon_error(Xte, Xte_pred, col_w)
    combo_te = df_test["combo_str"].astype(str).values
    thr_vec = np.array([thr_per_combo.get(c, thr_global) for c in combo_te], dtype=float)
    mask_recon = err_te >= thr_vec

    y_norm_pred = Xte_pred[:, j_y]
    amt_pred = invert_pred_to_inr(df_test, y_norm_pred, stats_train)
    amt_act  = df_test["amount"].astype(float).values

    mad_vec = np.array([stats_train.get(c, {}).get("mad_inr", 1.0) for c in combo_te], dtype=float)
    tol_abs = np.maximum.reduce([
        np.full_like(amt_act, ABS_TOL_INR, dtype=float),
        MAD_MULT * mad_vec,
        PCT_TOL * np.abs(amt_act)
    ])
    diff_abs = np.abs(amt_pred - amt_act)
    mask_inr = diff_abs >= tol_abs

    is_anom = mask_recon & mask_inr
    idx = np.where(is_anom)[0]

    anomalies = pd.DataFrame()
    if len(idx) > 0:
        out = df_test.iloc[idx].copy()
        out["amount_pred"] = amt_pred[idx]
        out["amount_diff_abs"] = diff_abs[idx]
        out["amount_diff_pct"] = out["amount_diff_abs"] / (np.abs(out["amount"]) + 1e-9)
        out["recon_error"] = err_te[idx]
        out["thr_recon"] = thr_vec[idx]
        out["is_anomaly"] = 1

        dev = compute_top_feature_deviations(
            X_true=Xte,
            X_pred=Xte_pred,
            col_names=col_names,
            j_y=j_y,
            anomaly_idx=idx,
            col_w=col_w,
            top_k=2,
        )
        for k, vals in dev.items():
            out[k] = vals

        out["reason"] = out.apply(format_reason, axis=1)

        keep = [
            "ts",
            "BankAccountCode", "BusinessUnitCode", "BankTransactionCode",
            "amount", "amount_pred", "amount_diff_abs", "amount_diff_pct",
            "recon_error", "thr_recon", "combo_str", "combo_id",
            "top_feat1_name", "top_feat1_actual", "top_feat1_recon",
            "top_feat2_name", "top_feat2_actual", "top_feat2_recon",
            "is_anomaly", "reason"
        ]
        anomalies = out.loc[:, [c for c in keep if c in out.columns]] \
                      .sort_values("recon_error", ascending=False) \
                      .reset_index(drop=True)

    # save artifacts
    anomalies.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)
    with open(os.path.join(OUT_DIR, COMBO_MAP_JSON), "w") as f:
        json.dump(make_combo_map_from_train(df_train), f, indent=2)

    meta = dict(
        mode="internal_split",
        total_train_rows=int(len(df_train)),
        total_valid_rows=int(len(df_valid)),
        total_test_rows=int(len(df_test)),
        total_anomalies=int(len(anomalies)),
        tabular_feature_cols=tab_cols,
        embed_dim=int(EMBED_DIM),
        enc_units=list(ENC_UNITS),
        dec_units=list(DEC_UNITS),
        weight_y_norm=float(WEIGHT_YNORM),
        weight_tabular=float(WEIGHT_TABULAR),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        min_samples_per_combo_threshold=int(MIN_SAMPLES_PER_COMBO_THR),
        abs_tol_inr=float(ABS_TOL_INR),
        pct_tol=float(PCT_TOL),
        mad_mult=float(MAD_MULT),
        model_path=os.path.join(OUT_DIR, MODEL_PATH),
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
        count_iqr_floor=float(COUNT_IQR_FLOOR),
        count_norm_clip_lo=float(COUNT_NORM_CLIP_LO),
        count_norm_clip_hi=float(COUNT_NORM_CLIP_HI),
        count_day_col=str(COUNT_DAY_COL),
    )
    with open(os.path.join(OUT_DIR, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done][internal] TRAIN={len(df_train)} VALID={len(df_valid)} TEST={len(df_test)} anomalies={len(anomalies)}")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")

    return anomalies, meta

# ====== external test pipeline (two DFs) ======
def run_pipeline_external_test_df(df_train_all: pd.DataFrame,
                                  df_test_all: pd.DataFrame,
                                  valid_frac: float = VALID_FRAC_IN_TRAIN) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build features from TRAIN and TEST DataFrames separately.
    Train/Valid from TRAIN (tail valid). TEST from TEST df. Train-only combo map/stats.
    Returns: (anomalies_df, meta_dict)
    """
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- FE on TRAIN df ---
    feats_train_all, tab_cols_train, _ = build_dataset_from_excel(df_train_all)
    tab_cols_train = [c for c in tab_cols_train if c not in IGNORE_FOR_RECON]
    df_train, df_valid = split_train_valid_tail_full(feats_train_all, ts_col="ts", valid_frac=valid_frac)

    # combo map from TRAIN
    combo_map = make_combo_map_from_train(df_train)
    df_train = apply_combo_map(df_train, combo_map)
    df_valid = apply_combo_map(df_valid, combo_map)

    # --- FE on TEST df ---
    feats_test_all, _, _ = build_dataset_from_excel(df_test_all)
    df_test = apply_combo_map(feats_test_all, combo_map)

    # Align tabular columns to TRAIN's set (after ignoring some)
    tab_cols = [c for c in tab_cols_train if c in df_test.columns]

    # 3) Per-combo stats (TRAIN ONLY)
    stats_train = build_combo_stats_train(df_train)

    # NEW: add count_norm to each split using TRAIN stats only (MUST be before X building)
    df_train = add_count_norm(df_train, stats_train)
    df_valid = add_count_norm(df_valid, stats_train)
    df_test  = add_count_norm(df_test,  stats_train)

    # ensure count_norm is part of AE tabular features (MUST be before X building)
    if "count_norm" not in tab_cols:
        tab_cols = tab_cols + ["count_norm"]

    # Save per-combo stats for inference
    combo_stats_path = os.path.join(OUT_DIR, COMBO_STATS_JSON)
    with open(combo_stats_path, "w") as f:
        json.dump(stats_to_jsonable(stats_train), f, indent=2)

    # 4) y_norm
    y_train = normalize_amount(df_train, stats_train)
    y_valid = normalize_amount(df_valid, stats_train)
    y_test  = normalize_amount(df_test,  stats_train)

    # 5) AE inputs (NOW includes count_norm)
    Xtr, Ctr, col_names, j_y = build_inputs_with_ynorm(df_train, tab_cols, y_train)
    Xva, Cva, _,   _         = build_inputs_with_ynorm(df_valid, tab_cols, y_valid)
    Xte, Cte, _,   _         = build_inputs_with_ynorm(df_test,  tab_cols, y_test)

    # 6) Weights (must use final tab_cols length)
    w_row_tr, col_w = build_sample_weights_for_recon(df_train, y_train, stats_train, n_tab=len(tab_cols), j_y=j_y)
    w_row_va, _     = build_sample_weights_for_recon(df_valid, y_valid, stats_train, n_tab=len(tab_cols), j_y=j_y)

    n_combos    = len(combo_map)
    n_accounts  = int(df_train["account_id"].max()) + 1
    n_busunits  = int(df_train["bu_id"].max()) + 1
    n_codes     = int(df_train["code_id"].max()) + 1

    # 7) Model
    model = make_autoencoder(
    n_in=Xtr.shape[1],
    n_combos=n_combos,
    n_accounts=n_accounts,
    n_busunits=n_busunits,
    n_codes=n_codes,
    embed_dim=EMBED_DIM,
    enc_units=ENC_UNITS,
    dec_units=DEC_UNITS,
    lr=LR,
    col_w=col_w
    )

    hist = model.fit(
        [Xtr, Ctr], Xtr,
        validation_data=([Xva, Cva], Xva, w_row_va),
        sample_weight=w_row_tr,
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, PATIENCE//3), min_lr=1e-5, verbose=1)
        ],
        verbose=1
    )
    model.save(os.path.join(OUT_DIR, MODEL_PATH), include_optimizer=False)
    plot_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # 8) Threshold from VALID
    Xva_pred = model.predict([Xva, Cva], batch_size=2048, verbose=0)
    err_va = row_recon_error(Xva, Xva_pred, col_w)
    thr_global = float(np.percentile(err_va, THRESHOLD_PERCENTILE))

    thr_per_combo: Dict[str, float] = {}
    df_thr = pd.DataFrame({"combo": df_valid["combo_str"].astype(str).values, "err": err_va})
    for c, g in df_thr.groupby("combo"):
        if len(g) >= MIN_SAMPLES_PER_COMBO_THR:
            thr_per_combo[c] = float(np.percentile(g["err"].values, THRESHOLD_PERCENTILE))

    # Save thresholds for inference
    thr_obj = thresholds_to_jsonable(thr_global, thr_per_combo)
    thresholds_path = os.path.join(OUT_DIR, THRESHOLDS_JSON)
    with open(thresholds_path, "w") as f:
        json.dump(thr_obj, f, indent=2)

    # 9) Score TEST (held-out)
    Xte_pred = model.predict([Xte, Cte], batch_size=2048, verbose=0)
    err_te = row_recon_error(Xte, Xte_pred, col_w)
    combo_te = df_test["combo_str"].astype(str).values
    thr_vec = np.array([thr_per_combo.get(c, thr_global) for c in combo_te], dtype=float)
    mask_recon = err_te >= thr_vec

    y_norm_pred = Xte_pred[:, j_y]
    amt_pred = invert_pred_to_inr(df_test, y_norm_pred, stats_train)
    amt_act  = df_test["amount"].astype(float).values

    mad_vec = np.array([stats_train.get(c, {}).get("mad_inr", 1.0) for c in combo_te], dtype=float)
    tol_abs = np.maximum.reduce([
        np.full_like(amt_act, ABS_TOL_INR, dtype=float),
        MAD_MULT * mad_vec,
        PCT_TOL * np.abs(amt_act)
    ])
    diff_abs = np.abs(amt_pred - amt_act)
    mask_inr = diff_abs >= tol_abs

    is_anom = mask_recon & mask_inr
    idx = np.where(is_anom)[0]

    anomalies = pd.DataFrame()
    if len(idx) > 0:
        out = df_test.iloc[idx].copy()
        out["amount_pred"] = amt_pred[idx]
        out["amount_diff_abs"] = diff_abs[idx]
        out["amount_diff_pct"] = out["amount_diff_abs"] / (np.abs(out["amount"]) + 1e-9)
        out["recon_error"] = err_te[idx]
        out["thr_recon"] = thr_vec[idx]
        out["is_anomaly"] = 1

        dev = compute_top_feature_deviations(
            X_true=Xte,
            X_pred=Xte_pred,
            col_names=col_names,
            j_y=j_y,
            anomaly_idx=idx,
            col_w=col_w,
            top_k=2,
        )
        for k, vals in dev.items():
            out[k] = vals

        out["reason"] = out.apply(format_reason, axis=1)

        keep = [
            "ts",
            "BankAccountCode", "BusinessUnitCode", "BankTransactionCode",
            "amount", "amount_pred", "amount_diff_abs", "amount_diff_pct",
            "recon_error", "thr_recon", "combo_str", "combo_id",
            "top_feat1_name", "top_feat1_actual", "top_feat1_recon",
            "top_feat2_name", "top_feat2_actual", "top_feat2_recon",
            "is_anomaly", "reason"
        ]
        anomalies = out.loc[:, [c for c in keep if c in out.columns]] \
                      .sort_values("recon_error", ascending=False) \
                      .reset_index(drop=True)

    # save artifacts
    anomalies.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)
    with open(os.path.join(OUT_DIR, COMBO_MAP_JSON), "w") as f:
        json.dump(make_combo_map_from_train(df_train), f, indent=2)

    meta = dict(
        mode="external_test",
        total_train_rows=int(len(df_train)),
        total_valid_rows=int(len(df_valid)),
        total_test_rows=int(len(df_test)),
        total_anomalies=int(len(anomalies)),
        tabular_feature_cols=[c for c in tab_cols],
        embed_dim=int(EMBED_DIM),
        enc_units=list(ENC_UNITS),
        dec_units=list(DEC_UNITS),
        weight_y_norm=float(WEIGHT_YNORM),
        weight_tabular=float(WEIGHT_TABULAR),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        min_samples_per_combo_threshold=int(MIN_SAMPLES_PER_COMBO_THR),
        abs_tol_inr=float(ABS_TOL_INR),
        pct_tol=float(PCT_TOL),
        mad_mult=float(MAD_MULT),
        model_path=os.path.join(OUT_DIR, MODEL_PATH),
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
        count_iqr_floor=float(COUNT_IQR_FLOOR),
        count_norm_clip_lo=float(COUNT_NORM_CLIP_LO),
        count_norm_clip_hi=float(COUNT_NORM_CLIP_HI),
        count_day_col=str(COUNT_DAY_COL),
    )
    with open(os.path.join(OUT_DIR, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done][external] TRAIN={len(df_train)} VALID={len(df_valid)} TEST={len(df_test)} anomalies={len(anomalies)}")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")

    return anomalies, meta


# ====== Convenience entry ======
def run_pipeline(df_all_or_train: pd.DataFrame,
                 df_test: Optional[pd.DataFrame] = None,
                 valid_frac: float = VALID_FRAC_IN_TRAIN) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    - Internal split: run_pipeline(df_all)
    - External test:  run_pipeline(df_train, df_test)
    """
    if df_test is None:
        return run_pipeline_internal_split_df(df_all_or_train, valid_frac=valid_frac)
    return run_pipeline_external_test_df(df_all_or_train, df_test, valid_frac=valid_frac)
