# training7_inline.py
# Autoencoder + dual aux heads for bank-statement features (after BinaryEncoder featureization).
# Train on ~first 10 months; score last ~2 months by 'ts'.
# Anomaly if:
#   (a) recon error >= per-(combo,magnitude-bucket) threshold, AND
#   (b) amount misprediction beyond adaptive INR tolerance (std & %) or log-space tolerance for tiny amounts.

import os, json, pickle, argparse, random, math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Silence TF logs and enable determinism-ish
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, regularizers, Model

# ====== bring your featureizer ======
from bank_features_training_xls import build_training_matrix_from_excel  # <- ensure this path/name

# =========================
# CONFIG
# =========================
OUT_DIR                    = "ae_outputs_final"
OUTPUT_CSV                 = "ae_anomalies_only.csv"
LEARNING_CURVE_PNG         = "learning_curve.png"
META_JSON                  = "ae_meta.json"
MODEL_PATH                 = "ae_model.keras"

# architecture / training
SIZES                      = (256, 128, 64)
L2                          = 0.0
DROPOUT                     = 0.0
LR                          = 1e-3
BATCH_SIZE                  = 512
EPOCHS                      = 200
PATIENCE                    = 20

# reconstruction thresholding
THRESHOLD_PERCENTILE        = 99.0
MIN_COMBO_SAMPLES_FOR_THR   = 30     # per-(combo, bucket) threshold minimum validation rows

# feature loss weighting for AE (not aux heads)
USE_COL_WEIGHT              = True
CAT_FEATURE_WEIGHT          = 0.20
NUM_FEATURE_WEIGHT          = 1.00
AMOUNT_FEATURE_WEIGHT       = 8.0
ZSCORE_FEATURE_WEIGHT       = 2.0

# amount gating (INR)
AMOUNT_DIFF_ABS             = 1.0     # base INR tolerance
AMOUNT_DIFF_PCT             = 0.05    # 5% relative tolerance
AMOUNT_DIFF_STD_MULT        = 0.5     # add k * combo std

# tiny-amount handling (log head)
USE_AUX_AMOUNT_HEAD_INR     = True    # predicts INR (MAE)
USE_AUX_AMOUNT_HEAD_LOG     = True    # predicts signed log1p(|amount|)
AMOUNT_AUX_LOSS_WEIGHT_INR  = 2.0
AMOUNT_AUX_LOSS_WEIGHT_LOG  = 3.0     # give more pull to smalls via log head
TINY_AMOUNT_SWITCH_INR      = 100.0   # <= this, prefer log-head pred over INR-head

# tiny-amount gating in log space
LOG_ABS_TOL                 = 0.25    # absolute tolerance in log space (~e^0.25 ≈ 1.28x)
# example: if actual=4, pred=6 -> log1p(4)=1.609, log1p(6)=1.946, diff~0.337 > 0.25 -> flag

# z-score gating
ZSCORE_AMOUNT_COMBO_THRESH  = 2.5     # set 0 to disable

# Artifacts from FE
SCALER_PATH                 = "artifacts_features/standard_scaler.pkl"
SCHEMA_PATH                 = "artifacts_features/feature_schema.json"

# Reproducibility
GLOBAL_SEED                 = 42

# =========================
# Utilities
# =========================

def set_global_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def enable_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def time_split_10m_train_2m_test(feats_df: pd.DataFrame, ts_col="ts") -> Tuple[np.ndarray, np.ndarray]:
    if ts_col not in feats_df.columns:
        raise ValueError(f"Expected '{ts_col}' in features dataframe.")
    df = feats_df.sort_values(ts_col).reset_index(drop=True).copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    max_ts = df[ts_col].max()
    cutoff_test_start = max_ts - pd.DateOffset(months=2)
    test_mask = df[ts_col] >= cutoff_test_start
    train_mask = ~test_mask
    return train_mask.values, test_mask.values

def train_valid_split_from_train_by_time(feats_df: pd.DataFrame,
                                         base_train_mask: np.ndarray,
                                         ts_col="ts",
                                         valid_frac_in_train=0.10) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.where(base_train_mask)[0]
    if len(idx) < 10:
        return base_train_mask.copy(), ~base_train_mask.copy()
    sub = feats_df.iloc[idx].sort_values(ts_col)
    n = len(sub)
    n_valid = max(1, int(round(n * valid_frac_in_train)))
    valid_idx = sub.index[-n_valid:]
    final_train_mask = base_train_mask.copy()
    final_valid_mask = np.zeros_like(base_train_mask, dtype=bool)
    final_valid_mask[valid_idx] = True
    final_train_mask[valid_idx] = False
    return final_train_mask, final_valid_mask

def load_scaler_schema() -> Tuple[Optional[Any], Dict]:
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return scaler, schema

def build_column_weights(feat_names: List[str], schema: Dict,
                         cat_weight=0.20, num_weight=1.0,
                         amount_feature_weight=8.0,
                         zscore_feature_weight=2.0) -> np.ndarray:
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols  = schema.get("passthrough_numeric_cols", [])
    num_names  = [f"{c}_scaled" for c in scale_cols] + pass_cols
    w = np.full(len(feat_names), cat_weight, dtype=np.float32)
    for i, n in enumerate(feat_names):
        if n in num_names: w[i] = num_weight
    if "amount" in scale_cols:
        amt_name = "amount_scaled"
        if amt_name in feat_names:
            w[feat_names.index(amt_name)] = amount_feature_weight
    if "zscore_amount_30d" in schema.get("numeric_cols", []):
        try:
            j = feat_names.index("zscore_amount_30d"); w[j] = max(w[j], zscore_feature_weight)
        except ValueError:
            pass
    return w

def make_weighted_huber(col_weights: np.ndarray, delta: float = 1.0):
    w = tf.constant(col_weights.reshape(1, -1), dtype=tf.float32)
    d = tf.constant(delta, dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        err = y_pred - y_true
        abs_err = tf.abs(err)
        hub = tf.where(abs_err <= d, 0.5 * tf.square(err), d * abs_err - 0.5 * d * d)
        hub_w = hub * w
        return tf.reduce_mean(tf.reduce_mean(hub_w, axis=1))
    return loss_fn

def make_autoencoder_with_dual_heads(input_dim: int,
                                     sizes=(256,128,64),
                                     l2=0.0, dropout=0.0, lr=1e-3) -> Model:
    inp = layers.Input(shape=(input_dim,), name="in")
    x = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(inp)
    if dropout>0: x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    bottleneck = layers.Dense(sizes[2], activation="relu", kernel_regularizer=regularizers.l2(l2), name="bottleneck")(x)

    # Decoder for reconstruction
    d = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(bottleneck)
    if dropout>0: d = layers.Dropout(dropout)(d)
    d = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(d)
    recon = layers.Dense(input_dim, activation=None, name="recon")(d)

    # Aux INR head
    if USE_AUX_AMOUNT_HEAD_INR:
        a_inr = layers.Dense(64, activation="relu")(bottleneck)
        amount_inr = layers.Dense(1, activation=None, name="amount_inr")(a_inr)
    else:
        amount_inr = None

    # Aux LOG head (signed log1p)
    if USE_AUX_AMOUNT_HEAD_LOG:
        a_log = layers.Dense(64, activation="relu")(bottleneck)
        amount_log = layers.Dense(1, activation=None, name="amount_log")(a_log)
    else:
        amount_log = None

    outputs = {"recon": recon}
    if amount_inr is not None: outputs["amount_inr"] = amount_inr
    if amount_log is not None: outputs["amount_log"] = amount_log

    model = Model(inputs=inp, outputs=outputs, name="ae_with_dual_amount_heads")
    opt = optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    losses_dict = {"recon": losses.Huber(delta=1.0)}
    loss_w = {"recon": 1.0}
    if amount_inr is not None:
        losses_dict["amount_inr"] = losses.MeanAbsoluteError()
        loss_w["amount_inr"] = AMOUNT_AUX_LOSS_WEIGHT_INR
    if amount_log is not None:
        losses_dict["amount_log"] = losses.MeanSquaredError()
        loss_w["amount_log"] = AMOUNT_AUX_LOSS_WEIGHT_LOG

    model.compile(optimizer=opt, loss=losses_dict, loss_weights=loss_w)
    return model

def ae_predict(model: Model, X: np.ndarray):
    out = model.predict(X, batch_size=2048, verbose=0)
    if isinstance(out, dict):
        return out.get("recon"), out.get("amount_inr"), out.get("amount_log")
    # single-output fallback
    return out, None, None

def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return ((x_true - x_pred) ** 2).mean(axis=1)

def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0) -> float:
    return float(np.percentile(valid_errors, percentile))

def plot_learning_curve(hist, out_path: str):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="train"); plt.plot(val_loss, label="valid")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ---------- Amount helpers ----------

def signed_log1p_amount(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))

def inv_signed_log1p_amount(l: np.ndarray) -> np.ndarray:
    return np.sign(l) * (np.expm1(np.abs(l)))

def magnitude_bucket(a: np.ndarray) -> np.ndarray:
    # floor(log10(|a|)) with special bucket for zeros/small
    eps = 1e-9
    ab = np.abs(a)
    buck = np.where(ab < eps, -9, np.floor(np.log10(ab + eps)).astype(int))
    # clamp to reasonable range
    buck = np.clip(buck, -9, 9)
    return buck

def find_amount_feature_index(feat_names: List[str], schema: Dict) -> Tuple[Optional[int], Optional[int]]:
    scale_cols = schema.get("scale_numeric_cols", [])
    if "amount" not in scale_cols: return None, None
    try: j_feat = feat_names.index("amount_scaled")
    except ValueError: return None, None
    j_slot = scale_cols.index("amount")
    return j_feat, j_slot

def invert_scale(scaled_val: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return scaled_val * scale + mean

# ---------- Amount-vs-Combo columns ----------

def resolve_combo_amount_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    candidates = [
        ("mean_amount_30d_by_combo", "std_amount_30d_by_combo", "zscore_amount_30d_by_combo"),
        ("mean_amount_30d_combo",    "std_amount_30d_combo",    "zscore_amount_30d_combo"),
        ("mean_amount_30d_acct_bu_code", "std_amount_30d_acct_bu_code", "zscore_amount_30d_acct_bu_code"),
    ]
    for mean_c, std_c, z_c in candidates:
        if mean_c in df.columns and std_c in df.columns and z_c in df.columns:
            return {"mean": mean_c, "std": std_c, "z": z_c}
    return {"mean": "mean_amount_30d", "std": "std_amount_30d", "z": "zscore_amount_30d"}

# ---------- Reasoning (INR-aware) ----------

def _fmt_inr(x) -> str:
    try: return f"₹{float(x):,.2f}"
    except Exception: return str(x)

def reason_amount_vs_combo(row: pd.Series, col_mean: str, col_std: str, col_z: str) -> str:
    amt_actual = row.get("amount_actual", row.get("AmountInBankAccountCurrency", row.get("amount", None)))
    amt_pred   = row.get("amount_pred", np.nan)

    acct = row.get("BankAccountCode", "-")
    bu   = row.get("BusinessUnitCode", "-")
    code = row.get("BankTransactionCode", "-")

    mu   = row.get(col_mean, None)
    sd   = row.get(col_std, None)
    z    = row.get(col_z, None)

    d_abs = row.get("amount_diff_abs", np.nan)
    d_pct = row.get("amount_diff_pct", np.nan)

    amt_txt   = _fmt_inr(amt_actual)
    pred_txt  = _fmt_inr(amt_pred) if pd.notna(amt_pred) else "—"
    mu_txt    = _fmt_inr(mu) if pd.notna(mu) else "their usual level"
    sd_txt    = _fmt_inr(sd) if (pd.notna(sd) and float(sd) != 0.0) else "—"
    d_abs_txt = _fmt_inr(d_abs) if pd.notna(d_abs) else "—"
    d_pct_txt = f"{float(d_pct)*100:.1f}%" if pd.notna(d_pct) else "—"
    z_txt     = f" (~{float(z):.2f}σ)" if pd.notna(z) else ""

    return (
        f"Amount {amt_txt} looks atypical for (Account='{acct}', BU='{bu}', Code='{code}'). "
        f"Pred ≈ {pred_txt}, Δ ≈ {d_abs_txt} ({d_pct_txt}). "
        f"Recent 30-day typical ≈ {mu_txt}, std ≈ {sd_txt}{z_txt}."
    )

# =========================
# main pipeline
# =========================
def run_pipeline(X_all: np.ndarray, feats_all: pd.DataFrame, feat_names: List[str]):
    set_global_seed()
    enable_gpu_memory_growth()

    os.makedirs(OUT_DIR, exist_ok=True)
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' in engineered features")

    # Split
    base_train_mask, test_mask = time_split_10m_train_2m_test(feats_all, ts_col="ts")
    train_mask, valid_mask = train_valid_split_from_train_by_time(feats_all, base_train_mask, ts_col="ts", valid_frac_in_train=0.10)

    X_train, X_valid, X_test = X_all[train_mask], X_all[valid_mask], X_all[test_mask]
    feats_train = feats_all.loc[train_mask].reset_index(drop=True)
    feats_valid = feats_all.loc[valid_mask].reset_index(drop=True)
    feats_test  = feats_all.loc[test_mask].reset_index(drop=True)

    # Column weights for AE
    scaler, schema = load_scaler_schema()
    loss_fn = None
    if USE_COL_WEIGHT and schema is not None:
        col_w = build_column_weights(
            feat_names, schema,
            cat_weight=CAT_FEATURE_WEIGHT,
            num_weight=NUM_FEATURE_WEIGHT,
            amount_feature_weight=AMOUNT_FEATURE_WEIGHT,
            zscore_feature_weight=ZSCORE_FEATURE_WEIGHT
        )
        loss_fn = make_weighted_huber(col_w, delta=1.0)

    # Aux targets and sample weights
    if "AmountInBankAccountCurrency" in feats_all.columns:
        y_amt_all = feats_all["AmountInBankAccountCurrency"].astype(float).values
    else:
        y_amt_all = feats_all["amount"].astype(float).values
    y_amt_train = y_amt_all[train_mask]
    y_amt_valid = y_amt_all[valid_mask]

    y_log_all = signed_log1p_amount(y_amt_all)
    y_log_train, y_log_valid = y_log_all[train_mask], y_log_all[valid_mask]

    # small amounts get higher weight
    w_train = 1.0 / (np.log1p(np.abs(y_amt_train)) + 1.0)
    w_valid = 1.0 / (np.log1p(np.abs(y_amt_valid)) + 1.0)
    # normalize weights
    w_train = w_train / (w_train.mean() + 1e-9)
    w_valid = w_valid / (w_valid.mean() + 1e-9)

    # Build model with dual heads
    ae = make_autoencoder_with_dual_heads(X_all.shape[1], sizes=SIZES, l2=L2, dropout=DROPOUT, lr=LR)

    # Prepare targets dict
    y_train = {"recon": X_train}
    y_valid = {"recon": X_valid}
    sw_train = {"recon": None}
    sw_valid = {"recon": None}
    if USE_AUX_AMOUNT_HEAD_INR:
        y_train["amount_inr"] = y_amt_train.reshape(-1, 1)
        y_valid["amount_inr"] = y_amt_valid.reshape(-1, 1)
        sw_train["amount_inr"] = w_train
        sw_valid["amount_inr"] = w_valid
    if USE_AUX_AMOUNT_HEAD_LOG:
        y_train["amount_log"] = y_log_train.reshape(-1, 1)
        y_valid["amount_log"] = y_log_valid.reshape(-1, 1)
        sw_train["amount_log"] = w_train
        sw_valid["amount_log"] = w_valid

    hist = ae.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid, sw_valid),
        sample_weight=sw_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, PATIENCE // 3), min_lr=1e-5, verbose=1)
        ],
        verbose=1
    )

    # Save & curve
    ae.save(os.path.join(OUT_DIR, MODEL_PATH), include_optimizer=False)
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # === Validation recon thresholds (per combo & bucket) ===
    pred_valid_recon, _, _ = ae_predict(ae, X_valid)
    valid_err = reconstruction_errors(X_valid, pred_valid_recon)
    thr_global = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # build keys
    if "combo_id" in feats_valid.columns:
        valid_combo = feats_valid["combo_id"].astype(str).values
    else:
        valid_combo = (feats_valid["BankAccountCode"].astype(str) + "|" +
                       feats_valid["BusinessUnitCode"].astype(str) + "|" +
                       feats_valid["BankTransactionCode"].astype(str)).values

    amt_valid = y_amt_valid
    buck_valid = magnitude_bucket(amt_valid)

    thr_per_key: Dict[Tuple[str, int], float] = {}
    df_thr = pd.DataFrame({"combo": valid_combo, "bucket": buck_valid, "err": valid_err})
    for (c, b), sub in df_thr.groupby(["combo", "bucket"]):
        if len(sub) >= MIN_COMBO_SAMPLES_FOR_THR:
            thr_per_key[(c, int(b))] = float(np.percentile(sub["err"].values, THRESHOLD_PERCENTILE))

    # === Score TEST ===
    pred_test_recon, pred_amt_inr, pred_amt_log = ae_predict(ae, X_test)
    test_err  = reconstruction_errors(X_test, pred_test_recon)

    if "combo_id" in feats_test.columns:
        test_combo = feats_test["combo_id"].astype(str).values
    else:
        test_combo = (feats_test["BankAccountCode"].astype(str) + "|" +
                      feats_test["BusinessUnitCode"].astype(str) + "|" +
                      feats_test["BankTransactionCode"].astype(str)).values
    amt_test_actual = feats_test.get("AmountInBankAccountCurrency", feats_test.get("amount")).astype(float).values
    buck_test = magnitude_bucket(amt_test_actual)

    thr_vec = np.array([thr_per_key.get((c, int(b)), thr_global) for c, b in zip(test_combo, buck_test)], dtype=float)
    is_anom = (test_err >= thr_vec)
    idx_anom = np.where(is_anom)[0]

    anomalies_only = pd.DataFrame()
    if len(idx_anom) > 0:
        out = feats_test.iloc[idx_anom].copy()
        out["recon_error"] = test_err[idx_anom]
        out["is_anomaly"] = 1

        # ===== Amount prediction (choose INR or LOG based on magnitude) =====
        # Fallback to inverse-scaling from recon if aux head missing
        # Prepare inverse-scaling objects
        j_feat, j_slot = find_amount_feature_index(feat_names, schema)
        scaler_mean = getattr(scaler, "mean_", None) if scaler is not None else None
        scaler_scale = getattr(scaler, "scale_", None) if scaler is not None else None

        # actual
        out["amount_actual"] = amt_test_actual[idx_anom]

        # preds from heads
        pred_inr = pred_amt_inr[idx_anom].reshape(-1) if pred_amt_inr is not None else None
        pred_log = pred_amt_log[idx_anom].reshape(-1) if pred_amt_log is not None else None
        pred_log_inr = inv_signed_log1p_amount(pred_log) if pred_log is not None else None

        # combine
        chosen_pred = np.empty(len(out), dtype=float)
        for i, a in enumerate(out["amount_actual"].values):
            if pred_inr is not None and pred_log_inr is not None:
                if abs(a) <= TINY_AMOUNT_SWITCH_INR:
                    chosen_pred[i] = pred_log_inr[i]
                else:
                    chosen_pred[i] = pred_inr[i]
            elif pred_inr is not None:
                chosen_pred[i] = pred_inr[i]
            elif pred_log_inr is not None:
                chosen_pred[i] = pred_log_inr[i]
            else:
                # inverse-scale from recon (fallback)
                if (j_feat is not None) and (j_slot is not None) and (scaler_mean is not None) and (scaler_scale is not None):
                    p_scaled = pred_test_recon[idx_anom, j_feat][i]
                    chosen_pred[i] = invert_scale(p_scaled, scaler_mean[j_slot], scaler_scale[j_slot])
                else:
                    chosen_pred[i] = np.nan

        out["amount_pred"] = chosen_pred
        out["amount_diff_abs"] = (out["amount_pred"] - out["amount_actual"]).abs()
        out["amount_diff_pct"] = out["amount_diff_abs"] / (np.abs(out["amount_actual"]) + 1e-9)

        # ===== Adaptive amount gating (INR) + optional log-space gate for tiny amounts =====
        combo_cols = resolve_combo_amount_cols(out)
        stdcol = combo_cols["std"]
        std_vals = out.get(stdcol, pd.Series(0.0, index=out.index)).fillna(0.0).astype(float).values
        tol_abs = np.maximum(AMOUNT_DIFF_ABS, AMOUNT_DIFF_STD_MULT * std_vals)

        mask_amount_inr = (out["amount_diff_abs"].values >= tol_abs) | (out["amount_diff_pct"].values >= AMOUNT_DIFF_PCT)

        # log-space gating for tiny amounts
        act_log = signed_log1p_amount(out["amount_actual"].values)
        pred_log_used = signed_log1p_amount(out["amount_pred"].values)
        mask_amount_log = np.abs(pred_log_used - act_log) >= LOG_ABS_TOL
        # only enforce log gate for tiny amounts (|actual| <= switch)
        tiny_mask = (np.abs(out["amount_actual"].values) <= TINY_AMOUNT_SWITCH_INR)
        mask_amount = np.where(tiny_mask, mask_amount_log, mask_amount_inr)

        # z-score gate
        if ZSCORE_AMOUNT_COMBO_THRESH > 0:
            zcol = combo_cols["z"]
            if zcol in out.columns:
                mask_z = np.abs(out[zcol].values) >= ZSCORE_AMOUNT_COMBO_THRESH
            else:
                mask_z = np.ones(len(out), dtype=bool)
        else:
            mask_z = np.ones(len(out), dtype=bool)

        out = out.loc[mask_amount & mask_z].copy()

        # Reasoning
        if not out.empty:
            out["reason"] = out.apply(lambda r: reason_amount_vs_combo(r, combo_cols["mean"], combo_cols["std"], combo_cols["z"]), axis=1)

        # Order & save
        keep_cols = [
            "ts",
            "BankAccountCode", "BusinessUnitCode", "BankTransactionCode",
            "AmountInBankAccountCurrency", "amount_actual", "amount_pred",
            "amount_diff_abs", "amount_diff_pct",
            combo_cols.get("mean"), combo_cols.get("std"), combo_cols.get("z"),
            "recon_error", "is_anomaly", "reason"
        ]
        keep_cols = [c for c in keep_cols if c in out.columns]
        anomalies_only = out.loc[:, keep_cols].sort_values(["recon_error"], ascending=False).reset_index(drop=True)

    # Save artifacts
    os.makedirs(OUT_DIR, exist_ok=True)
    anomalies_only.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)

    # Metadata
    meta = dict(
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        total_train_rows=int(len(feats_train)),
        total_valid_rows=int(len(feats_valid)),
        total_scored_rows=int(len(feats_test)),
        total_anomalies=int(len(anomalies_only)),
        feature_names=feat_names,
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
        loss_weighting=bool(USE_COL_WEIGHT),
        weights=dict(cat=float(CAT_FEATURE_WEIGHT), num=float(NUM_FEATURE_WEIGHT),
                     amount=float(AMOUNT_FEATURE_WEIGHT), zscore=float(ZSCORE_FEATURE_WEIGHT)),
        split="train=first~10m, test=last~2m (valid tail of train)",
        zscore_amount_combo_thresh=float(ZSCORE_AMOUNT_COMBO_THRESH),
        amount_diff_abs=float(AMOUNT_DIFF_ABS),
        amount_diff_pct=float(AMOUNT_DIFF_PCT),
        amount_diff_std_mult=float(AMOUNT_DIFF_STD_MULT),
        use_aux_amount_head_inr=bool(USE_AUX_AMOUNT_HEAD_INR),
        use_aux_amount_head_log=bool(USE_AUX_AMOUNT_HEAD_LOG),
        aux_loss_weights=dict(inr=float(AMOUNT_AUX_LOSS_WEIGHT_INR), log=float(AMOUNT_AUX_LOSS_WEIGHT_LOG)),
        tiny_amount_switch_inr=float(TINY_AMOUNT_SWITCH_INR),
        log_abs_tol=float(LOG_ABS_TOL),
        model_path=os.path.join(OUT_DIR, MODEL_PATH),
        per_combo_bucket_threshold_min_samples=int(MIN_COMBO_SAMPLES_FOR_THR),
    )
    with open(os.path.join(OUT_DIR, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] TRAIN={len(feats_train)} VALID={len(feats_valid)} TEST={len(feats_test)} "
          f"anomalies={len(anomalies_only)} (per-combo+bucket thresholds)")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")
    if len(anomalies_only) > 0:
        print(anomalies_only.head(5).to_string(index=False))

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to history Excel used for training + scoring")
    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index (default 0)")
    args = parser.parse_args()

    X_all, feats_all, feat_names = build_training_matrix_from_excel(args.excel, sheet_name=args.sheet)
    run_pipeline(X_all, feats_all, feat_names)

if __name__ == "__main__":
    main()
