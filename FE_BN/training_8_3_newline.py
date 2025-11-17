# training7_inline.py
# Autoencoder for bank-statement features (after BinaryEncoder featureization).
# Trains on ~first 10 months; scores last ~2 months by 'ts'.
# Flags ONLY rows where:
#   (a) overall recon error is high  AND
#   (b) predicted 'amount' differs from actual (beyond tolerance) for the SAME (Account, BU, TxnCode) combo
# Includes inverse-scaling to show reasons in ORIGINAL INR units.

import os, json, pickle, argparse, random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow deterministic-ish setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, regularizers, Model

# ====== bring your featureizer ======
# Must provide: build_training_matrix_from_excel(history_xlsx_path, sheet_name=0)
# and persist artifacts into artifacts_features/ (scaler, schema, baselines, encoder)
from bank_features_training_xls import build_training_matrix_from_excel  # adjust filename if needed

# =========================
# CONFIG
# =========================
OUT_DIR                 = "ae_outputs_final"
OUTPUT_CSV              = "ae_anomalies_only.csv"
LEARNING_CURVE_PNG      = "learning_curve.png"
META_JSON               = "ae_meta.json"
MODEL_PATH              = "ae_model.keras"

# model/training
SIZES                   = (256, 128, 64)
L2                      = 0.0
DROPOUT                 = 0.0
LR                      = 1e-3
BATCH_SIZE              = 512
EPOCHS                  = 200
PATIENCE                = 20
THRESHOLD_PERCENTILE    = 99.0

# loss weighting (helps balance numeric vs encoded-binary bits)
USE_COL_WEIGHT          = True
CAT_FEATURE_WEIGHT      = 0.25
NUM_FEATURE_WEIGHT      = 1.00

# Amount gating (in ORIGINAL currency units)
AMOUNT_DIFF_ABS         = 1.0        # ₹ absolute tolerance; set to 0 for any nonzero diff
AMOUNT_DIFF_PCT         = 0.05       # 5% relative tolerance (on |actual|); set to 0 to disable

# Restrict to clearly unusual combos too (zscore from FE)
ZSCORE_AMOUNT_COMBO_THRESH = 2.5      # set 0 to disable this gating

# Artifacts persisted by FE script
SCALER_PATH             = "artifacts_features/standard_scaler.pkl"
SCHEMA_PATH             = "artifacts_features/feature_schema.json"

# Reproducibility
GLOBAL_SEED             = 42

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
    """Boolean masks: train (~first 10 months) & test (last ~2 months) by 'ts'."""
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
    """Within the training slice, carve out a small time-ordered validation tail."""
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

# ---------- Amount helpers ----------

def find_amount_feature_index(feat_names: List[str], schema: Dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (amount_scaled_index_in_feat_names, slot_index_in_scaler_stats)
    """
    scale_cols = schema.get("scale_numeric_cols", [])
    if "amount" not in scale_cols:
        return None, None
    target_feat = "amount_scaled"
    try:
        j_feat = feat_names.index(target_feat)
    except ValueError:
        return None, None
    j_slot = scale_cols.index("amount")
    return j_feat, j_slot

def invert_scale(scaled_val: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return scaled_val * scale + mean

# ---------- Amount-vs-Combo z-score columns resolver ----------

def resolve_combo_amount_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Prefer combo-level FE columns; fallback to base names if not present.
    """
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
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return str(x)

def reason_amount_vs_combo(row: pd.Series, col_mean: str, col_std: str, col_z: str) -> str:
    # Prefer inverse-scaled columns computed in this script
    amt_actual = row.get("amount_actual", row.get("AmountInBankAccountCurrency", row.get("amount", None)))
    amt_pred   = row.get("amount_pred", np.nan)

    acct = row.get("BankAccountCode", "-")
    bu   = row.get("BusinessUnitCode", "-")
    code = row.get("BankTransactionCode", "-")

    mu   = row.get(col_mean, None)   # FE columns are in original units
    sd   = row.get(col_std, None)
    z    = row.get(col_z, None)

    d_abs = row.get("amount_diff_abs", np.nan)
    d_pct = row.get("amount_diff_pct", np.nan)

    # Pretty bits
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

    # Split 10m/2m
    base_train_mask, test_mask = time_split_10m_train_2m_test(feats_all, ts_col="ts")
    train_mask, valid_mask = train_valid_split_from_train_by_time(
        feats_all, base_train_mask, ts_col="ts", valid_frac_in_train=0.10
    )
    X_train, X_valid, X_test = X_all[train_mask], X_all[valid_mask], X_all[test_mask]
    feats_valid = feats_all.loc[valid_mask].reset_index(drop=True)
    feats_test  = feats_all.loc[test_mask].reset_index(drop=True)

    # Loss weighting
    scaler, schema = load_scaler_schema()
    loss_fn = None
    if USE_COL_WEIGHT and schema is not None:
        col_w = build_column_weights(feat_names, schema,
                                     cat_weight=CAT_FEATURE_WEIGHT,
                                     num_weight=NUM_FEATURE_WEIGHT)
        loss_fn = make_weighted_huber(col_w, delta=1.0)

    # Train AE
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
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=max(3, PATIENCE // 3), min_lr=1e-5, verbose=1)
        ],
        verbose=1
    )
    # Save model and learning curve
    ae.save(os.path.join(OUT_DIR, MODEL_PATH), include_optimizer=False)
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # Threshold from VALID
    pred_valid = ae_predict(ae, X_valid)
    valid_err = reconstruction_errors(X_valid, pred_valid)
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # Score TEST (last 2 months)
    pred_test = ae_predict(ae, X_test)
    test_err  = reconstruction_errors(X_test, pred_test)

    # Base anomaly by recon error
    is_anom   = (test_err >= thr)
    idx_anom  = np.where(is_anom)[0]

    anomalies_only = pd.DataFrame()
    if len(idx_anom) > 0:
        out = feats_test.iloc[idx_anom].copy()
        out["recon_error"] = test_err[idx_anom]
        out["is_anomaly"]  = 1

        # === Gate by amount misprediction on SAME combo ===
        # find amount index in features and invert-scale to original INR
        j_feat, j_slot = find_amount_feature_index(feat_names, schema)
        scaler_mean = getattr(scaler, "mean_", None) if scaler is not None else None
        scaler_scale = getattr(scaler, "scale_", None) if scaler is not None else None

        if (j_feat is not None) and (j_slot is not None) and (scaler_mean is not None) and (scaler_scale is not None):
            a_scaled = X_test[idx_anom, j_feat]
            p_scaled = pred_test[idx_anom, j_feat]
            a_amt = invert_scale(a_scaled, scaler_mean[j_slot], scaler_scale[j_slot])
            p_amt = invert_scale(p_scaled, scaler_mean[j_slot], scaler_scale[j_slot])

            out["amount_pred"] = p_amt
            if "AmountInBankAccountCurrency" in out.columns:
                out["amount_actual"] = out["AmountInBankAccountCurrency"].astype(float)
            else:
                out["amount_actual"] = out["amount"].astype(float)

            out["amount_diff_abs"] = (out["amount_pred"] - out["amount_actual"]).abs()
            out["amount_diff_pct"] = (out["amount_diff_abs"] / (out["amount_actual"].abs() + 1e-9))

            mask_amount_gate = (out["amount_diff_abs"] >= AMOUNT_DIFF_ABS) | (out["amount_diff_pct"] >= AMOUNT_DIFF_PCT)
        else:
            # If we cannot confidently invert-scale amount, do NOT gate out by amount diff
            mask_amount_gate = np.ones(len(out), dtype=bool)

        # === Optional z-score gate from FE (combo-level preferred) ===
        if ZSCORE_AMOUNT_COMBO_THRESH > 0:
            combo_cols = resolve_combo_amount_cols(out)
            zcol = combo_cols["z"]
            if zcol in out.columns:
                mask_z = out[zcol].abs() >= ZSCORE_AMOUNT_COMBO_THRESH
            else:
                mask_z = np.ones(len(out), dtype=bool)
        else:
            combo_cols = resolve_combo_amount_cols(out)
            mask_z = np.ones(len(out), dtype=bool)

        out = out.loc[mask_amount_gate & mask_z].copy()

        # English reasoning (INR-aware)
        if not out.empty:
            out["reason"] = out.apply(
                lambda r: reason_amount_vs_combo(r, combo_cols["mean"], combo_cols["std"], combo_cols["z"]),
                axis=1
            )

        # Nice ordering
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

    # Save
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
        split="train=first~10m, test=last~2m (valid is tail of train)",
        zscore_amount_combo_thresh=float(ZSCORE_AMOUNT_COMBO_THRESH),
        amount_diff_abs=float(AMOUNT_DIFF_ABS),
        amount_diff_pct=float(AMOUNT_DIFF_PCT),
        model_path=os.path.join(OUT_DIR, MODEL_PATH),
    )
    with open(os.path.join(OUT_DIR, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] TEST rows={len(feats_test)} anomalies={len(anomalies_only)} thr={thr:.6f}")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")
    if len(anomalies_only) > 0:
        print(anomalies_only.head(5).to_string(index=False))


# =========================
# CLI Usage (optional)
# =========================
# Example:
#   python training7_inline.py --excel bank_history.xlsx --sheet 0
# Will build features, split 10m/2m, train AE, and write outputs under OUT_DIR.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to history Excel used for training + scoring")
    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index (default 0)")
    args = parser.parse_args()

    X_all, feats_all, feat_names = build_training_matrix_from_excel(args.excel, sheet_name=args.sheet)
    run_pipeline(X_all, feats_all, feat_names)

if __name__ == "__main__":
    main()
