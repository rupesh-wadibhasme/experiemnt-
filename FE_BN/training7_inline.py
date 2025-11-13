# training7.py
# Autoencoder for bank-statement features after BinaryEncoder featureization.
# - Works with FE_BN_6_inline.py (BinaryEncoder for all categoricals)
# - Trains on first ~10 months; scores last ~2 months by 'ts'
# - Top-K contributor reporting (inverse scaling for numeric top feature)

import os, json, pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, regularizers, Model

# ====== bring your featureizer ======
# Rename your file to FE_BN_6_inline.py (no parentheses) and make sure
# it exposes build_training_matrix_from_excel + artifacts in artifacts_features/
from FE_BN_6_inline import build_training_matrix_from_excel

# =========================
# CONFIG
# =========================
HISTORY_PATH = "bank_history_3yrs.xlsx"   # used only if you call build_training_matrix_from_excel here
SHEET_NAME   = 0

OUT_DIR      = "ae_outputs_final"
OUTPUT_CSV   = "ae_anomalies_only.csv"
LEARNING_CURVE_PNG = "learning_curve.png"
META_JSON    = "ae_meta.json"

# model/training
SIZES        = (256, 128, 64)
L2           = 0.0
DROPOUT      = 0.0
LR           = 1e-3
BATCH_SIZE   = 512
EPOCHS       = 200
PATIENCE     = 20
THRESHOLD_PERCENTILE = 99.0

# loss weighting (helps balance numeric vs encoded-binary bits)
USE_COL_WEIGHT       = True
CAT_FEATURE_WEIGHT   = 0.25
NUM_FEATURE_WEIGHT   = 1.00

# Top-K contributors
TOP_K = 3

# artifacts from FE script
SCALER_PATH          = "artifacts_features/standard_scaler.pkl"
SCHEMA_PATH          = "artifacts_features/feature_schema.json"

# =========================
# Utilities
# =========================

def time_split_10m_train_2m_test(feats_df: pd.DataFrame, ts_col="ts") -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks: train (~first 10 months) & test (last ~2 months) by 'ts'."""
    if ts_col not in feats_df.columns:
        raise ValueError(f"Expected '{ts_col}' in features dataframe.")
    df = feats_df.sort_values(ts_col).reset_index(drop=True).copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    max_ts = df[ts_col].max()
    cutoff_test_start = max_ts - pd.DateOffset(months=2)  # last ~2 months for scoring
    test_mask = df[ts_col] >= cutoff_test_start
    train_mask = ~test_mask
    return train_mask.values, test_mask.values

def train_valid_split_from_train_by_time(feats_df: pd.DataFrame,
                                         base_train_mask: np.ndarray,
                                         ts_col="ts",
                                         valid_frac_in_train=0.10) -> Tuple[np.ndarray, np.ndarray]:
    """Within the training slice, carve out a small time-ordered validation tail."""
    idx = np.where(base_train_mask)[0]
    if len(idx) < 10:
        # edge-case: tiny data → use the same as both train/valid
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
                         cat_weight=0.25, num_weight=1.0) -> np.ndarray:
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols  = schema.get("passthrough_numeric_cols", [])
    num_names  = [f"{c}_scaled" for c in scale_cols] + pass_cols
    w = np.full(len(feat_names), cat_weight, dtype=np.float32)
    for i, n in enumerate(feat_names):
        if n in num_names:
            w[i] = num_weight
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

def make_autoencoder(input_dim: int,
                     sizes=(256, 128, 64),
                     l2=0.0,
                     dropout=0.0,
                     lr=1e-3,
                     use_layernorm=True,
                     loss_fn=None) -> Model:
    inp = layers.Input(shape=(input_dim,), name="in")
    x = inp
    if use_layernorm:
        x = layers.LayerNormalization(axis=-1, name="in_norm")(x)
    x = layers.Dense(sizes[0], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    if dropout > 0: x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[1], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    bottleneck = layers.Dense(sizes[2], activation="relu",
                              kernel_regularizer=regularizers.l2(l2),
                              name="bottleneck")(x)
    x = layers.Dense(sizes[1], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(bottleneck)
    if dropout > 0: x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[0], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    out = layers.Dense(input_dim, activation=None, name="recon")(x)
    ae = Model(inputs=inp, outputs=out, name="dense_autoencoder")
    opt = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    ae.compile(optimizer=opt, loss=loss_fn if loss_fn is not None else losses.Huber(delta=1.0))
    return ae

def ae_predict(ae_model: Model, X: np.ndarray) -> np.ndarray:
    return ae_model.predict(X, batch_size=2048, verbose=0)

def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return ((x_true - x_pred) ** 2).mean(axis=1)

def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0) -> float:
    return float(np.percentile(valid_errors, percentile))

def plot_learning_curve(hist, out_path: str):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="train loss"); plt.plot(val_loss, label="valid loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch"); plt.ylabel("Huber"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ---------- Top-K contributors (BinaryEncoder friendly) ----------

def topk_by_feature(X: np.ndarray,
                    P: np.ndarray,
                    feat_names: List[str],
                    schema: Dict,
                    scaler) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Returns:
      idx_topk: (N, K) indices into feat_names
      feat_names_top1: list of top-1 feature name per row
      top1_actual_orig: numeric inverse-scaled actual (only for numeric & if scaled), else NaN
      top1_pred_orig:   numeric inverse-scaled pred   (only for numeric & if scaled), else NaN
    Notes:
      - With BinaryEncoder, categorical bits are just separate columns (no inverse decode).
      - We still report feature names; for numeric scaled features we inverse scale the top-1.
    """
    sq = (X - P) ** 2                    # per-feature squared error
    K = min(TOP_K, sq.shape[1])
    idx_topk = np.argpartition(-sq, kth=K-1, axis=1)[:, :K]
    # order each row
    row_vals = np.take_along_axis(sq, idx_topk, axis=1)
    order = np.argsort(-row_vals, axis=1)
    idx_topk = np.take_along_axis(idx_topk, order, axis=1)

    # inverse only for numeric_scaled features on TOP-1
    scale_cols = schema.get("scale_numeric_cols", [])
    scaler_mean = getattr(scaler, "mean_", None) if scaler is not None else None
    scaler_scale = getattr(scaler, "scale_", None) if scaler is not None else None
    scale_slot = {f"{c}_scaled": j for j, c in enumerate(scale_cols)}

    feat_names_top1 = []
    top1_actual_orig = np.full(X.shape[0], np.nan, dtype=np.float32)
    top1_pred_orig   = np.full(X.shape[0], np.nan, dtype=np.float32)

    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    for r in range(X.shape[0]):
        j = idx_topk[r, 0]
        fname = feat_names[j]
        feat_names_top1.append(fname)

        # inverse only if it's a scaled numeric feature
        if fname.endswith("_scaled") and (scaler_mean is not None) and (scaler_scale is not None):
            slot = scale_slot.get(fname, None)
            if slot is not None and slot < len(scaler_mean):
                a_s, p_s = X[r, j], P[r, j]
                a_o = float(a_s * scaler_scale[slot] + scaler_mean[slot])
                p_o = float(p_s * scaler_scale[slot] + scaler_mean[slot])
                top1_actual_orig[r] = a_o
                top1_pred_orig[r]   = p_o

    return idx_topk, feat_names_top1, top1_actual_orig, top1_pred_orig
                        
# ---- Reasoning config (amount vs account+BU) ----
ZSCORE_AMOUNT_THRESH = 2.5  # adjust: 2.0 more sensitive, 3.0 stricter

def _fmt_inr(x) -> str:
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return str(x)

def _reason_amount_vs_combo(row) -> str:
    """
    English sentence focused on amount anomaly for (BankAccountCode, BusinessUnitCode).
    Expects these columns on the row:
      - 'AmountInBankAccountCurrency', 'BankAccountCode', 'BusinessUnitCode'
      - 'mean_amount_30d', 'std_amount_30d', 'zscore_amount_30d'
    """
    acct = row.get("BankAccountCode", "-")
    bu   = row.get("BusinessUnitCode", "-")
    amt  = row.get("AmountInBankAccountCurrency", row.get("amount", None))
    mu   = row.get("mean_amount_30d", None)
    sd   = row.get("std_amount_30d", None)
    z    = row.get("zscore_amount_30d", None)

    # defensive fallbacks
    amt_txt = _fmt_inr(amt)
    mu_txt  = _fmt_inr(mu) if pd.notna(mu) else "their usual level"
    sd_txt  = _fmt_inr(sd) if (pd.notna(sd) and float(sd) != 0.0) else "—"

    z_txt = ""
    try:
        if pd.notna(z):
            z_txt = f" (~{float(z):.2f}σ)"
    except Exception:
        pass

    return (
        f"Amount {amt_txt} looks atypical for Account '{acct}' and BU '{bu}': "
        f"recent 30-day typical ≈ {mu_txt}, std ≈ {sd_txt}{z_txt}."
    )

# =========================
# main pipeline
# =========================
def run_pipeline(X_all: np.ndarray, feats_all: pd.DataFrame, feat_names: List[str]):
    os.makedirs(OUT_DIR, exist_ok=True)
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' in engineered features")

    # ---- Split: first ~10 months (train) / last ~2 months (test) ----
    base_train_mask, test_mask = time_split_10m_train_2m_test(feats_all, ts_col="ts")
    # carve out small validation from the training period (time-ordered tail)
    train_mask, valid_mask = train_valid_split_from_train_by_time(feats_all, base_train_mask, ts_col="ts", valid_frac_in_train=0.10)

    X_train, X_valid, X_test = X_all[train_mask], X_all[valid_mask], X_all[test_mask]
    feats_test = feats_all.loc[test_mask].reset_index(drop=True)

    # ---- Loss weighting
    scaler, schema = load_scaler_schema()
    loss_fn = None
    if USE_COL_WEIGHT and schema is not None:
        col_w = build_column_weights(feat_names, schema,
                                     cat_weight=CAT_FEATURE_WEIGHT,
                                     num_weight=NUM_FEATURE_WEIGHT)
        loss_fn = make_weighted_huber(col_w, delta=1.0)

    # ---- Train AE
    input_dim = X_all.shape[1]
    ae = make_autoencoder(input_dim,
                          sizes=SIZES, l2=L2, dropout=DROPOUT,
                          lr=LR, use_layernorm=True, loss_fn=loss_fn)
    hist = ae.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, PATIENCE // 3),
                                        min_lr=1e-5, verbose=1)
        ],
        verbose=1
    )
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # ---- Threshold from VALID
    pred_valid = ae_predict(ae, X_valid)
    valid_err = reconstruction_errors(X_valid, pred_valid)
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # ---- Score TEST (last 2 months)
    pred_test = ae_predict(ae, X_test)
    test_err  = reconstruction_errors(X_test, pred_test)
    is_anom   = (test_err >= thr)
    idx_anom  = np.where(is_anom)[0]

    # ---- Top-K contributors on TEST anomalies
    anomalies_only = pd.DataFrame()
    if len(idx_anom) > 0:
        idx_topk, top1_names, top1_act, top1_pred = topk_by_feature(
            X_test[idx_anom], pred_test[idx_anom], feat_names, schema, scaler
        )
        # flatten top-k names into strings
        topk_names_str = []
        for row in idx_topk:
            topk_names_str.append(", ".join([feat_names[j] for j in row]))
    
        out = feats_test.iloc[idx_anom].copy()
        out["recon_error"] = test_err[idx_anom]
        out["is_anomaly"]  = 1
        out["topk_features"] = topk_names_str
        out["top1_feature"]  = top1_names
        out["top1_actual_orig"] = top1_act
        out["top1_pred_orig"]   = top1_pred
    
        # === Keep ONLY amount-vs-(Account,BU) anomalies ===
        # Require engineered 30d z-score to be sufficiently large
        if "zscore_amount_30d" in out.columns:
            mask_amt = out["zscore_amount_30d"].abs() >= ZSCORE_AMOUNT_THRESH
        else:
            # if for some reason missing, keep all (but this should exist from FE)
            mask_amt = np.ones(len(out), dtype=bool)
    
        out = out.loc[mask_amt].copy()
    
        # Attach single-scenario English reasoning
        # (amount atypical for this Account + BU)
        if not out.empty:
            out["reason"] = out.apply(_reason_amount_vs_combo, axis=1)
    
        anomalies_only = out.sort_values("recon_error", ascending=False).reset_index(drop=True)

    # Save CSV (empty file if no anomalies survived the amount filter)
    os.makedirs(OUT_DIR, exist_ok=True)
    anomalies_only.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)


    meta = dict(
        threshold=float(thr),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        total_scored_rows=int(len(feats_test)),
        total_anomalies=int(len(anomalies_only)),
        feature_names=feat_names,
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
        loss_weighting=bool(USE_COL_WEIGHT),
        top_k=int(TOP_K),
        split="train=first~10m, test=last~2m (valid is tail of train)"
    )
    with open(os.path.join(OUT_DIR, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] TEST rows={len(feats_test)} anomalies={len(anomalies_only)} thr={thr:.6f}")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")


# =========================
# Quick entry point example
# =========================
if __name__ == "__main__":
    # If you prefer building here:
    # X_all, feats_all, feat_names = build_training_matrix_from_excel(HISTORY_PATH, sheet_name=SHEET_NAME)
    # run_pipeline(X_all, feats_all, feat_names)
    raise SystemExit("Import run_pipeline(...) and call with (X_all, feats_all, feat_names).")
