# ae_keras_full_detect.py
# Autoencoder for bank-statement features with hybrid-encoded categoricals (vectorized Top-K contributors)
# - robust training: Huber + grad clipping + ReduceLROnPlateau + EarlyStopping
# - threshold from validation (p99)
# - vectorized group-wise Top-K contributors (numeric / low-card OHE / high-card hashed)
# - human-readable reasons WITHOUT error value; numerics show original-unit actual/pred
# - outputs Top-3 features per row

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
SIZES        = (256, 128, 64)
L2           = 0.0
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

# Top-K contributors
TOP_K = 3   # number of contributors to report

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


# ====== Vectorized Top-K contributor logic ======

def build_index_maps(feat_names: List[str], schema: Dict, encoder: Any):
    """
    Returns:
      num_idx            : list[int] numeric feature indices (scaled+passthrough)
      num_is_scaled      : list[bool] aligned with num_idx (True if '<col>_scaled')
      num_scale_slot     : list[int or None] scaler slot for scaled numerics
      low_groups         : dict col -> list[int]  (indices of OHE bits)
      high_groups        : dict col -> list[int]  (indices of hash bins)
      low_categories     : dict col -> list[str]  (labels for decoding OHE argmax)
    """
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols  = schema.get("passthrough_numeric_cols", [])
    scaled_names = [f"{c}_scaled" for c in scale_cols]
    num_names  = scaled_names + pass_cols
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    num_idx = [name_to_idx[n] for n in num_names if n in name_to_idx]
    num_is_scaled = [n in scaled_names for n in num_names if n in name_to_idx]
    # map '<col>_scaled' -> scaler slot index
    scale_slot = {f"{c}_scaled": j for j, c in enumerate(scale_cols)}
    num_scale_slot = [scale_slot.get(n, None) for n in num_names if n in name_to_idx]

    meta = schema.get("hybrid", {})
    low_cols  = meta.get("low_card_cols", [])
    high_cols = meta.get("high_card_cols", [])

    def pick(prefix: str, is_hash: bool):
        if is_hash:
            return [i for i, n in enumerate(feat_names) if n.startswith(f"{prefix}__hash_")]
        else:
            return [i for i, n in enumerate(feat_names) if n.startswith(f"{prefix}_")]

    low_groups  = {c: pick(c, is_hash=False) for c in low_cols}
    high_groups = {c: pick(c, is_hash=True)  for c in high_cols}

    low_categories = {}
    if getattr(encoder, "onehot", None) is not None:
        for c, cats in zip(encoder.low_card_cols, encoder.onehot.categories_):
            low_categories[c] = [str(x) for x in cats]

    return (num_idx, num_is_scaled, num_scale_slot,
            low_groups, high_groups, low_categories)


def vectorized_group_errors(X: np.ndarray,
                            P: np.ndarray,
                            feat_names: List[str],
                            schema: Dict,
                            encoder: Any):
    """
    Compute group errors for all rows:
      - numeric groups: single-index squared error
      - low OHE groups: sum over group's indices
      - high hash groups: sum over group's indices
    Returns:
      group_names [G], group_kinds [G], group_indices [G list], errors [N x G]
    """
    sq = (X - P) ** 2
    (num_idx, _num_is_scaled, _num_scale_slot,
     low_groups, high_groups, _low_categories) = build_index_maps(feat_names, schema, encoder)

    group_names: List[str] = []
    group_kinds: List[str] = []
    group_indices: List[List[int]] = []

    # numerics: one group per feature index
    for i in num_idx:
        group_names.append(feat_names[i])
        group_kinds.append("numeric")
        group_indices.append([i])

    # low-card OHE
    for col, idxs in low_groups.items():
        if idxs:
            group_names.append(col)
            group_kinds.append("lowcat")
            group_indices.append(idxs)

    # high-card hashed
    for col, idxs in high_groups.items():
        if idxs:
            group_names.append(col)
            group_kinds.append("highhash")
            group_indices.append(idxs)

    # build error matrix (N x G)
    N = X.shape[0]
    G = len(group_names)
    errs = np.zeros((N, G), dtype=np.float32)

    # fill numeric groups
    g_ptr = 0
    for i in num_idx:
        errs[:, g_ptr] = sq[:, i]
        g_ptr += 1

    # fill lowcat/highhash groups
    for col, idxs in low_groups.items():
        if idxs:
            errs[:, g_ptr] = np.sum(sq[:, idxs], axis=1)
            g_ptr += 1
    for col, idxs in high_groups.items():
        if idxs:
            errs[:, g_ptr] = np.sum(sq[:, idxs], axis=1)
            g_ptr += 1

    return group_names, group_kinds, group_indices, errs


def pick_topk_groups(errs: np.ndarray, k: int) -> np.ndarray:
    """
    Fast Top-K indices per row by error magnitude.
    Returns indices array of shape (N, k) sorted desc by error.
    """
    k = min(k, errs.shape[1])
    idx = np.argpartition(-errs, kth=k-1, axis=1)[:, :k]
    # sort those k by actual value desc
    row_vals = np.take_along_axis(errs, idx, axis=1)
    order = np.argsort(-row_vals, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def decode_lowcat_preds(P: np.ndarray,
                        group_indices: List[List[int]],
                        group_names: List[str],
                        group_kinds: List[str],
                        low_categories: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    For each lowcat group, compute argmax over reconstructed group -> predicted label string.
    Returns dict col -> np.ndarray[str] of length N.
    """
    preds = {}
    for gname, gkind, idxs in zip(group_names, group_kinds, group_indices):
        if gkind != "lowcat":
            continue
        # argmax index (local)
        local_idx = np.argmax(P[:, idxs], axis=1)
        cats = low_categories.get(gname, [])
        # map to string labels (unknown -> None)
        pred_labels = np.array([cats[i] if i < len(cats) else None for i in local_idx], dtype=object)
        preds[gname] = pred_labels
    return preds


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


# ====== Text reasons WITHOUT error values ======

def make_reason_numeric(name: str, actual_orig, pred_orig) -> str:
    return f"{name} was unusual (actual={actual_orig:.4f}, reconstructed≈{pred_orig:.4f})"

def make_reason_lowcat(col: str, actual_label: Any, pred_label: Any) -> str:
    if col in ("BankAccountCode", "BusinessUnitCode"):
        return f"{col} looked atypical for this context (input='{actual_label}', reconstructed as '{pred_label}')"
    return f"{col} category mismatch (input='{actual_label}', reconstructed='{pred_label}')"

def make_reason_highhash(col: str, actual_label: Any) -> str:
    if col in ("BankAccountCode", "BusinessUnitCode"):
        return f"{col} token '{actual_label}' looked rare/atypical versus learned patterns"
    return f"{col} token '{actual_label}' looked rare/unusual"


# =========================
# main pipeline
# =========================
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

    # 3) loss weighting
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

    # 4b) learning curve
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # 5) threshold from VALID
    pred_valid = ae_predict(ae, X_valid)
    valid_err = reconstruction_errors(X_valid, pred_valid)
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # 6) score ENTIRE dataset
    pred_all = ae_predict(ae, X_all)
    all_err = reconstruction_errors(X_all, pred_all)

    # 7) VECTORIZED Top-K contributors
    (num_idx, num_is_scaled, num_scale_slot,
     low_groups, high_groups, low_categories) = build_index_maps(feat_names, schema, encoder)

    # Build group errors
    group_names, group_kinds, group_indices, errs = vectorized_group_errors(
        X_all, pred_all, feat_names, schema, encoder
    )

    # Top-K indices per row
    topk_idx = pick_topk_groups(errs, TOP_K)  # shape (N, K)
    N, K = topk_idx.shape

    # Precompute decoded lowcat predicted labels (vectorized)
    lowcat_pred_labels = decode_lowcat_preds(pred_all, group_indices, group_names, group_kinds, low_categories)

    # For numeric inverse-scaling: maps
    scale_cols = schema.get("scale_numeric_cols", [])
    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    scale_slot = {f"{c}_scaled": j for j, c in enumerate(scale_cols)}
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    # Prepare output columns
    top_features = []
    top_feature_kinds = []
    reasons = []
    # Optional compatibility columns (first top numeric actual/pred)
    top_feature_actual_scaled = np.full(N, np.nan, dtype=np.float32)
    top_feature_pred_scaled   = np.full(N, np.nan, dtype=np.float32)

    # Build reasons per row (only K small strings per row; unavoidable Python loop but fast)
    for r in range(N):
        names_r = []
        kinds_r = []
        reasons_r = []

        for j in range(K):
            g = topk_idx[r, j]
            gname = group_names[g]
            gkind = group_kinds[g]
            idxs  = group_indices[g]

            if gkind == "numeric":
                i_feat = idxs[0]
                fname = feat_names[i_feat]
                # inverse to original units if scaled
                if fname.endswith("_scaled") and scaler_mean is not None and scaler_scale is not None:
                    slot = scale_slot.get(fname, None)
                    a_s = X_all[r, i_feat]; p_s = pred_all[r, i_feat]
                    if slot is not None and slot < len(scaler_mean):
                        a_o = float(a_s * scaler_scale[slot] + scaler_mean[slot])
                        p_o = float(p_s * scaler_scale[slot] + scaler_mean[slot])
                    else:
                        a_o = float(X_all[r, i_feat])
                        p_o = float(pred_all[r, i_feat])
                else:
                    a_o = float(X_all[r, i_feat])
                    p_o = float(pred_all[r, i_feat])

                names_r.append(fname)
                kinds_r.append("numeric")
                reasons_r.append(make_reason_numeric(fname, a_o, p_o))

                # Fill compatibility columns for the first Top-1 only
                if j == 0:
                    top_feature_actual_scaled[r] = X_all[r, i_feat]
                    top_feature_pred_scaled[r]   = pred_all[r, i_feat]

            elif gkind == "lowcat":
                # actual from feats_all; predicted from lowcat_pred_labels
                act = str(feats_all.iloc[r][gname]) if gname in feats_all.columns else None
                pred = lowcat_pred_labels.get(gname, np.array([None]*N))[r]
                names_r.append(gname)
                kinds_r.append("lowcat")
                reasons_r.append(make_reason_lowcat(gname, act, pred))

            else:  # highhash
                act = str(feats_all.iloc[r][gname]) if gname in feats_all.columns else None
                names_r.append(gname)
                kinds_r.append("highhash")
                reasons_r.append(make_reason_highhash(gname, act))

        top_features.append(", ".join(names_r))
        top_feature_kinds.append(", ".join(kinds_r))
        reasons.append(" | ".join(reasons_r))

    # 8) attach to full DF
    feats_all_out = feats_all.copy()
    feats_all_out["recon_error"] = all_err
    feats_all_out["is_anomaly"] = (all_err >= thr).astype(int)
    feats_all_out["top_features"] = top_features            # comma-separated Top-K names
    feats_all_out["top_feature_kinds"] = top_feature_kinds  # comma-separated kinds aligned to names
    feats_all_out["top_feature_actual_scaled"] = top_feature_actual_scaled  # only Top-1 numeric, kept for compat
    feats_all_out["top_feature_pred_scaled"]   = top_feature_pred_scaled
    feats_all_out["contrib_reason"] = reasons               # " | "-joined Top-K reason strings

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
        top_k=int(TOP_K),
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
