# ae_keras_full_detect.py
# Autoencoder for bank-statement features (simple, end-to-end)
# - build features
# - time-based split -> train / valid
# - train AE (Keras) with EarlyStopping
# - SAVE learning curve
# - learn threshold from VALID (p99)
# - score ENTIRE dataset
# - keep ONLY rows above threshold
# - for each anomaly: which feature deviated most, actual/pred (scaled), best-effort original, human reason

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, Model

#from bank_features_training_xls import build_training_matrix_from_excel  # your featureizer

# =========================
# CONFIG
# =========================
HISTORY_PATH = "bank_history_3yrs.xlsx"
OUT_DIR      = "ae_outputs_final"
OUTPUT_CSV   = "ae_anomalies_only.csv"          # we save ONLY anomalies
LEARNING_CURVE_PNG = "learning_curve.png"       # plot will be saved here

SHEET_NAME   = 0
ONE_HOT      = True

# split
CUTOFF       = None       # e.g. "2025-06-30"; if None -> use VALID_FRAC
VALID_FRAC   = 0.15

# model/training
SIZES        = (128, 64, 32)
L2           = 1e-6
DROPOUT      = 0.0
LR           = 1e-3
BATCH_SIZE   = 512
EPOCHS       = 5
PATIENCE     = 8
THRESHOLD_PERCENTILE = 99.0

# optional scaler for inverse-transform
SCALER_PATH  = "artifacts_features/standard_scaler.pkl"   # if not found, we'll keep orig=None


# =========================
# utilities
# =========================

def ae_predict_with_snapping(ae_model, X, feat_names,
                             int_like_feats=("month", "quarter", "day_of_week", "day_of_month")):
    """
    AE predict + postprocess:
    - run model.predict
    - for each row, snap int-like features to nearest valid int
    NOTE: this assumes those int-like features were NOT z-scored/minmaxed.
    """
    preds = ae_model.predict(X, batch_size=2048, verbose=0)
    # snapped = preds.copy()
    # for r in range(snapped.shape[0]):
    #     for i, name in enumerate(feat_names):
    #         base = name.split("_", 1)[0]
    #         if base in int_like_feats:
    #             v = float(snapped[r, i])
    #             if base == "month":
    #                 v = round(v); v = max(1, min(12, v))
    #             elif base == "quarter":
    #                 v = round(v); v = max(1, min(4, v))
    #             elif base == "day_of_week":
    #                 v = round(v); v = max(0, min(6, v))   # 0=Mon..6=Sun
    #             elif base == "day_of_month":
    #                 v = round(v); v = max(1, min(31, v))
    #             snapped[r, i] = v
    return preds


def time_based_split(feats_df: pd.DataFrame, ts_col="ts", cutoff=None, valid_frac=0.15):
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


def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return ((x_true - x_pred) ** 2).mean(axis=1)


def make_autoencoder(input_dim: int,
                     sizes=(128, 64, 32),
                     l2=1e-6,
                     dropout=0.0,
                     lr=1e-3) -> Model:
    inp = layers.Input(shape=(input_dim,), name="in")
    x = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    bottleneck = layers.Dense(sizes[2], activation="relu", kernel_regularizer=regularizers.l2(l2),
                              name="bottleneck")(x)
    x = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(bottleneck)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    out = layers.Dense(input_dim, activation=None, name="recon")(x)

    ae = Model(inputs=inp, outputs=out, name="dense_autoencoder")
    ae.compile(optimizer=optimizers.Adam(learning_rate=lr),
               loss=losses.MeanSquaredError())
    return ae


def train_autoencoder(ae: Model,
                      X_train: np.ndarray,
                      X_valid: np.ndarray,
                      batch_size=512,
                      epochs=200,
                      patience=15):
    es = callbacks.EarlyStopping(monitor="val_loss",
                                 patience=patience,
                                 restore_best_weights=True,
                                 verbose=1)
    hist = ae.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[es],
        verbose=1
    )
    return hist, ae


def plot_learning_curve(hist, out_path: str):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(7, 4))
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="valid loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0):
    return float(np.percentile(valid_errors, percentile))


def load_scaler_if_any(path: str):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def inverse_single(scaler, feat_idx: int, val: float):
    if scaler is None:
        return None
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        return None
    if feat_idx >= len(scaler.mean_):
        return None
    return val * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]


def make_reason_from_feature(feat_name: str,
                             err_val: float,
                             actual_orig,
                             pred_orig):
    # calendar-like
    if feat_name == "month":
        return (
            f"transaction happened in unusual month (actual={actual_orig}, usually {pred_orig}) "
            f"(err={err_val:.4f})"
        )
    if feat_name == "day_of_week":
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        act = int(round(actual_orig)) if actual_orig is not None else None
        pred = int(round(pred_orig)) if pred_orig is not None else None
        act_txt = weekdays[act] if act is not None and 0 <= act < 7 else actual_orig
        pred_txt = weekdays[pred] if pred is not None and 0 <= pred < 7 else pred_orig
        return (
            f"transaction happened on unusual weekday (actual={act_txt}, usual {pred_txt}) "
            f"(err={err_val:.4f})"
        )
    if feat_name == "day_of_month":
        return (
            f"day of month was unusual (actual={actual_orig}, usually {pred_orig}) "
            f"(err={err_val:.4f})"
        )
    if feat_name == "quarter":
        return (
            f"transaction quarter was unusual (actual={actual_orig}, usually {pred_orig}) "
            f"(err={err_val:.4f})"
        )

    # z-score style
    if feat_name == "zscore_amount_30d":
        return (
            "amount deviated strongly from this account's 30-day pattern "
            f"(err={err_val:.4f})"
        )

    # general helpers
    nicenames = {
        "amount": "amount was unusual",
        "amount_log": "amount (log) was unusual",
        "amount_sign": "transaction direction was unusual",
        "posting_lag_days": "posting lag was unusual",
        "same_amount_count_per_day": "same-amount count today was unusual",
        "txn_count_7d": "7-day txn volume was unusual",
        "txn_count_30d": "30-day txn volume was unusual",
        "mean_amount_30d": "30-day average amount was unusual",
        "std_amount_30d": "30-day volatility was unusual",
        "has_matched_ref": "matched reference looked unusual",
        "cashbook_flag": "cashbook flag looked unusual",
    }
    if feat_name in nicenames:
        extra = ""
        if actual_orig is not None and pred_orig is not None:
            extra = f" (actual={actual_orig}, predicted≈{pred_orig})"
        return f"{nicenames[feat_name]}{extra} (err={err_val:.4f})"

    # one-hot-ish
    if "_" in feat_name:
        prefix, rest = feat_name.split("_", 1)
        if prefix in ("BankAccountCode", "BankTransactionCode", "BusinessUnitCode", "CashBookFlag"):
            return f"{prefix}='{rest}' contributed most (err={err_val:.4f})"

    return f"{feat_name} contributed most (err={err_val:.4f})"


# =========================
# main pipeline
# =========================
def run_pipeline(X_all, feats_all, feat_names):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) features
    # X_all, feats_all, feat_names = build_training_matrix_from_excel(
    #     HISTORY_PATH, sheet_name=SHEET_NAME, one_hot=ONE_HOT
    # )
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' in engineered features")

    # 2) split
    train_mask, valid_mask = time_based_split(
        feats_all, ts_col="ts", cutoff=CUTOFF, valid_frac=VALID_FRAC
    )
    X_train, X_valid = X_all[train_mask], X_all[valid_mask]

    # 3) train AE
    input_dim = X_all.shape[1]
    ae = make_autoencoder(input_dim, sizes=SIZES, l2=L2, dropout=DROPOUT, lr=LR)
    hist, ae = train_autoencoder(ae, X_train, X_valid,
                                 batch_size=BATCH_SIZE,
                                 epochs=EPOCHS,
                                 patience=PATIENCE)

    # 3b) save learning curve
    plot_learning_curve(hist, os.path.join(OUT_DIR, LEARNING_CURVE_PNG))

    # 4) score VALID to get threshold
    #pred_valid = ae.predict(X_valid, batch_size=2048, verbose=0)
    pred_valid = ae_predict_with_snapping(ae,X_valid, feat_names)
    valid_err = reconstruction_errors(X_valid, pred_valid)
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # 5) score ENTIRE dataset
    #pred_all = ae.predict(X_all, batch_size=2048, verbose=0)
    pred_all = ae_predict_with_snapping(ae,X_valid, feat_names)
    all_err = reconstruction_errors(X_all, pred_all)

    # 6) per-feature deviation
    all_sqerr = (X_all - pred_all) ** 2
    all_max_idx = np.argmax(all_sqerr, axis=1)
    all_top_feat = [feat_names[i] for i in all_max_idx]
    all_top_feat_err = all_sqerr[np.arange(len(all_max_idx)), all_max_idx]
    all_top_actual_scaled = X_all[np.arange(len(all_max_idx)), all_max_idx]
    all_top_pred_scaled   = pred_all[np.arange(len(all_max_idx)), all_max_idx]

    # 7) inverse scaling (best effort)
    scaler = load_scaler_if_any(SCALER_PATH)
    all_top_actual_orig, all_top_pred_orig = [], []
    for idx, a_s, p_s in zip(all_max_idx, all_top_actual_scaled, all_top_pred_scaled):
        a_o = inverse_single(scaler, idx, a_s)
        p_o = inverse_single(scaler, idx, p_s)
        all_top_actual_orig.append(a_o)
        all_top_pred_orig.append(p_o)

    # 8) build reasons
    all_reasons = [
        make_reason_from_feature(f, e, a_o, p_o)
        for f, e, a_o, p_o in zip(
            all_top_feat, all_top_feat_err, all_top_actual_orig, all_top_pred_orig
        )
    ]

    # 9) attach to full DF
    feats_all_out = feats_all.copy()
    feats_all_out["recon_error"] = all_err
    feats_all_out["is_anomaly"] = (all_err >= thr).astype(int)
    feats_all_out["top_feature"] = all_top_feat
    feats_all_out["top_feature_error"] = all_top_feat_err
    feats_all_out["top_feature_actual_scaled"] = all_top_actual_scaled
    feats_all_out["top_feature_pred_scaled"] = all_top_pred_scaled
    feats_all_out["top_feature_actual_orig"] = all_top_actual_orig
    feats_all_out["top_feature_pred_orig"] = all_top_pred_orig
    feats_all_out["contrib_reason"] = all_reasons

    # 10) SAVE ONLY ANOMALIES
    anomalies_only = (
        feats_all_out[feats_all_out["is_anomaly"] == 1]
        .sort_values("recon_error", ascending=False)
        .reset_index(drop=True)
    )
    anomalies_only.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)

    # save meta
    meta = dict(
        threshold=float(thr),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        total_rows=int(len(feats_all_out)),
        total_anomalies=int(len(anomalies_only)),
        feature_names=feat_names,
        learning_curve=os.path.join(OUT_DIR, LEARNING_CURVE_PNG),
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
    run_pipeline(X_train, feats_train, feat_names)
