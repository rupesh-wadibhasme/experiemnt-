# ae_keras_full_detect.py
# Autoencoder for bank-statement features with hybrid-encoded categoricals
# - builds features via bank_features_training_xls (HybridEncoder)
# - robust training (Huber + grad clipping + ReduceLROnPlateau + EarlyStopping)
# - threshold from validation (p99)
# - anomaly scoring for entire dataset
# - group-wise top contributor (numeric vs low-card one-hot vs high-card hashed)
# - human-readable reasons (decode categorical labels when possible; numerics show actual vs predicted)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, Dict, List, Any

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, Model

# ====== bring your featureizer ======
from bank_features_training_xls import (
    build_training_matrix_from_excel,   # returns (X, feats_df, feat_names)
)

# =========================
# CONFIG
# =========================
HISTORY_PATH = "bank_history_3yrs.xlsx"
OUT_DIR      = "ae_outputs_final"
OUTPUT_CSV   = "ae_anomalies_only.csv"
LEARNING_CURVE_PNG = "learning_curve.png"

SHEET_NAME   = 0

# split
CUTOFF       = None       # e.g., "2025-06-30"; if None -> use VALID_FRAC
VALID_FRAC   = 0.30

# model/training
SIZES        = (256, 128, 64)   # a bit wider works better with OHE/hashed inputs
L2           = 0.0              # keep 0 while iterating on scaling; add back later if needed
DROPOUT      = 0.0
LR           = 1e-3
BATCH_SIZE   = 512
EPOCHS       = 200
PATIENCE     = 20
THRESHOLD_PERCENTILE = 99.0

# loss weighting (helps balance numeric vs categorical blocks)
USE_COL_WEIGHT       = True
CAT_FEATURE_WEIGHT   = 0.25
NUM_FEATURE_WEIGHT   = 1.00

# artifacts from featureizer
SCALER_PATH          = "artifacts_features/standard_scaler.pkl"
SCHEMA_PATH          = "artifacts_features/feature_schema.json"
ENCODER_PATH         = "artifacts_features/encoder.pkl"   # HybridEncoder pickle


# =========================
# utilities
# =========================

def time_based_split(feats_df: pd.DataFrame, ts_col="ts", cutoff=None, valid_frac=0.30):
    df = feats_df.sort_values(ts_col).reset_index(drop=True)
    if cutoff:
        cut = pd.to_datetime(cutoff)
        train_mask = df[ts_col] < cut
        valid_mask = ~train_mask
    else:
        n = len(df)
        n_train = int(np.floor((1 - valid_frac) * n))
        train_mask = pd.Series([True]*n_train + [False]*(n - n_train))
        valid_mask = ~train_mask
    return train_mask.values, valid_mask.values


def ae_predict(ae_model: Model, X: np.ndarray) -> np.ndarray:
    return ae_model.predict(X, batch_size=2048, verbose=0)


def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    # mean squared error across features
    return ((x_true - x_pred) ** 2).mean(axis=1)


def plot_learning_curve(hist, out_path: str):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="valid loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0):
    return float(np.percentile(valid_errors, percentile))


def load_scaler_schema_encoder() -> Tuple[Any, Dict, Any]:
    scaler = None
    schema = None
    encoder = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)   # HybridEncoder
    return scaler, schema, encoder


# ====== Loss & Model ======

def make_weighted_huber(col_weights: np.ndarray, delta: float = 1.0):
    """Column-weighted Huber loss to balance numerics vs categorical bits."""
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
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[1], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    bottleneck = layers.Dense(sizes[2], activation="relu",
                              kernel_regularizer=regularizers.l2(l2),
                              name="bottleneck")(x)
    x = layers.Dense(sizes[1], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(bottleneck)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[0], activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    out = layers.Dense(input_dim, activation=None, name="recon")(x)

    ae = Model(inputs=inp, outputs=out, name="dense_autoencoder")

    opt = optimizers.Adam(learning_rate=lr, clipnorm=1.0)  # gradient clipping
    if loss_fn is None:
        loss_fn = losses.Huber(delta=1.0)  # robust default
    ae.compile(optimizer=opt, loss=loss_fn)
    return ae


def train_autoencoder(ae: Model,
                      X_train: np.ndarray,
                      X_valid: np.ndarray,
                      batch_size=512,
                      epochs=200,
                      patience=20):
    es = callbacks.EarlyStopping(monitor="val_loss",
                                 patience=patience,
                                 restore_best_weights=True,
                                 verbose=1)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                        factor=0.5,
                                        patience=max(3, patience // 3),
                                        min_lr=1e-5,
                                        verbose=1)
    hist = ae.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[es, rlrop],
        verbose=1
    )
    return hist, ae


# ====== Contributor logic ======

def build_index_maps(feat_names: List[str], schema: Dict, encoder: Any):
    """
    Identify numeric indices and group categorical columns:
      - numeric: scaled cols -> names '<col>_scaled'; passthrough numerics -> their original names
      - low-card one-hot group indices for each categorical column
      - high-card hashed group indices for each categorical column
    """
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols  = schema.get("passthrough_numeric_cols", [])
    num_names  = [f"{c}_scaled" for c in scale_cols] + pass_cols
    num_idx    = [i for i, n in enumerate(feat_names) if n in num_names]

    meta = schema.get("hybrid", {})
    low_cols  = meta.get("low_card_cols", [])
    high_cols = meta.get("high_card_cols", [])

    # helper to pick feature indices by prefix
    def pick(prefix: str, is_hash: bool):
        if is_hash:
            return [i for i, n in enumerate(feat_names) if n.startswith(f"{prefix}__hash_")]
        else:
            # OneHotEncoder: "<col>_<label>"
            return [i for i, n in enumerate(feat_names) if n.startswith(f"{prefix}_")]

    low_groups  = {c: pick(c, is_hash=False) for c in low_cols}
    high_groups = {c: pick(c, is_hash=True)  for c in high_cols}

    # categories for low-card columns (for decoding predicted argmax)
    low_categories = {}
    if getattr(encoder, "onehot", None) is not None:
        for c, cats in zip(encoder.low_card_cols, encoder.onehot.categories_):
            low_categories[c] = [str(x) for x in cats]

    return num_idx, low_groups, high_groups, low_categories


def per_row_top_contributor(X: np.ndarray,
                            P: np.ndarray,
                            r: int,
                            feat_names: List[str],
                            schema: Dict,
                            encoder: Any,
                            feats_df_row: pd.Series):
    """
    For row r, compute top contributor:
      - numerics: max per-feature squared error
      - low-card OHE: sum squared error across group; report input label + reconstructed label
      - high-card hashed: sum squared error across bins; report input label only
    Returns: (group_name, kind, payload_dict)
    """
    num_idx, low_groups, high_groups, low_categories = build_index_maps(feat_names, schema, encoder)

    sq = (X - P) ** 2

    best_name, best_kind, best_payload, best_err = None, None, None, -1.0

    # numeric (per feature)
    for i in num_idx:
        e = float(sq[r, i])
        if e > best_err:
            best_err = e
            best_name = feat_names[i]
            best_kind = "numeric"
            best_payload = {
                "col": feat_names[i],
                "err": e,
                "actual_scaled": float(X[r, i]),
                "pred_scaled":   float(P[r, i]),
            }

    # low-card OHE groups
    for col, idxs in low_groups.items():
        if not idxs:
            continue
        e = float(np.sum(sq[r, idxs]))
        if e > best_err:
            # decode predicted label via argmax over reconstructed group
            pred_idx_local = int(np.argmax(P[r, idxs]))
            cats = low_categories.get(col, [])
            pred_label = cats[pred_idx_local] if pred_idx_local < len(cats) else None

            # actual label from original feats row (more reliable than X one-hot)
            actual_label = str(feats_df_row[col]) if col in feats_df_row else None

            best_err = e
            best_name = col
            best_kind = "lowcat"
            best_payload = {
                "col": col,
                "err": e,
                "actual_label": actual_label,
                "pred_label": pred_label,
            }

    # high-card hashed groups
    for col, idxs in high_groups.items():
        if not idxs:
            continue
        e = float(np.sum(sq[r, idxs]))
        if e > best_err:
            actual_label = str(feats_df_row[col]) if col in feats_df_row else None
            best_err = e
            best_name = col
            best_kind = "highhash"
            best_payload = {
                "col": col,
                "err": e,
                "actual_label": actual_label,
                "pred_label": None,   # not decodable from hash
            }

    return best_name, best_kind, best_payload


def build_column_weights(feat_names: List[str], schema: Dict,
                         cat_weight=0.25, num_weight=1.0) -> np.ndarray:
    """Make a vector of per-feature weights for the loss function."""
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols  = schema.get("passthrough_numeric_cols", [])
    num_names  = [f"{c}_scaled" for c in scale_cols] + pass_cols

    w = np.full(len(feat_names), cat_weight, dtype=np.float32)
    for i, n in enumerate(feat_names):
        if n in num_names:
            w[i] = num_weight
    return w


# ====== Reason text ======

def reason_from_payload(name: str, kind: str, payload: Dict) -> str:
    e = payload.get("err", None)
    if kind == "numeric":
        a = payload.get("actual_scaled")
        p = payload.get("pred_scaled")
        return f"{name} was unusual (actual={a:.4f}, reconstructed≈{p:.4f}) (err={e:.4f})"

    if kind == "lowcat":
        a = payload.get("actual_label")
        p = payload.get("pred_label")
        if name in ("BankAccountCode", "BusinessUnitCode"):
            return (f"{name} looked atypical for this context (input='{a}', model reconstructed as '{p}') "
                    f"(err={e:.4f})")
        return f"{name} category mismatch (input='{a}', reconstructed='{p}') (err={e:.4f})"

    if kind == "highhash":
        a = payload.get("actual_label")
        if name in ("BankAccountCode", "BusinessUnitCode"):
            return (f"{name} token '{a}' looked rare/atypical versus learned patterns (err={e:.4f})")
        return f"{name} token '{a}' looked rare/unusual (err={e:.4f})"

    # fallback
    return f"{name} contributed most (err={e:.4f})"


# ====== Main pipeline ======

def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) features
    X_all, feats_all, feat_names = build_training_matrix_from_excel(
        HISTORY_PATH, sheet_name=SHEET_NAME
    )
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' in engineered features")

    # 2) split
    train_mask, valid_mask = time_based_split(
        feats_all, ts_col="ts", cutoff=CUTOFF, valid_frac=VALID_FRAC
    )
    X_train, X_valid = X_all[train_mask], X_all[valid_mask]

    # 3) loss weighting (optional)
    scaler, schema, encoder = load_scaler_schema_encoder()
    loss_fn = None
    if USE_COL_WEIGHT and schema is not None:
        col_w = build_column_weights(feat_names, schema,
                                     cat_weight=CAT_FEATURE_WEIGHT,
                                     num_weight=NUM_FEATURE_WEIGHT)
        loss_fn = make_weighted_huber(col_w, delta=1.0)

    # 4) train AE
    input_dim = X_all.shape[1]
    ae = make_autoencoder(input_dim,
                          sizes=SIZES,
                          l2=L2,
                          dropout=DROPOUT,
                          lr=LR,
                          use_layernorm=True,
                          loss_fn=loss_fn)
    hist, ae = train_autoencoder(ae, X_train, X_valid,
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS,
                                 patience=PATIENCE)

    # 4b) save learning curve
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # 5) threshold from VALID
    pred_valid = ae_predict(ae, X_valid)
    valid_err = reconstruction_errors(X_valid, pred_valid)
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # 6) score ENTIRE dataset
    pred_all = ae_predict(ae, X_all)
    all_err = reconstruction_errors(X_all, pred_all)

    # 7) top contributor per row (group-wise) + reason
    top_name = []
    top_kind = []
    top_err  = []
    # for CSV compatibility: only fill these when the top feature is numeric
    top_actual_scaled = []
    top_pred_scaled   = []
    reasons = []

    for r in range(X_all.shape[0]):
        name, kind, payload = per_row_top_contributor(
            X_all, pred_all, r, feat_names, schema, encoder, feats_all.iloc[r]
        )
        top_name.append(name)
        top_kind.append(kind)
        top_err.append(payload.get("err", np.nan))

        if kind == "numeric":
            top_actual_scaled.append(payload.get("actual_scaled", np.nan))
            top_pred_scaled.append(payload.get("pred_scaled", np.nan))
        else:
            top_actual_scaled.append(np.nan)
            top_pred_scaled.append(np.nan)

        reasons.append(reason_from_payload(name, kind, payload))

    # 8) attach to full DF
    feats_all_out = feats_all.copy()
    feats_all_out["recon_error"] = all_err
    feats_all_out["is_anomaly"] = (all_err >= thr).astype(int)
    feats_all_out["top_feature"] = top_name
    feats_all_out["top_feature_kind"] = top_kind
    feats_all_out["top_feature_error"] = top_err
    feats_all_out["top_feature_actual_scaled"] = top_actual_scaled
    feats_all_out["top_feature_pred_scaled"]   = top_pred_scaled
    feats_all_out["contrib_reason"] = reasons

    # 9) SAVE ONLY ANOMALIES
    anomalies_only = (
        feats_all_out[feats_all_out["is_anomaly"] == 1]
        .sort_values("recon_error", ascending=False)
        .reset_index(drop=True)
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    anomalies_only.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)

    # 10) save meta
    meta = dict(
        threshold=float(thr),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        total_rows=int(len(feats_all_out)),
        total_anomalies=int(len(anomalies_only)),
        feature_names=feat_names,
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
        loss_weighting=bool(USE_COL_WEIGHT),
    )
    with open(os.path.join(OUT_DIR, "ae_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] rows={len(feats_all_out)} anomalies={len(anomalies_only)} thr={thr:.6f}")
    print(f"saved anomalies: {os.path.join(OUT_DIR, OUTPUT_CSV)}")
    print(f"learning curve: {os.path.join(OUT_DIR, LEARNING_CURVE_PNG)}")


# =========================
# run
# =========================
if __name__ == "__main__":
    run_pipeline()
