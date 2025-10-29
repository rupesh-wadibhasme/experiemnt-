# ae_keras_fit_plain.py
# Keras Autoencoder for bank-statement features (function-only, simple).
# - Builds matrix via bank_features_training_xls.py
# - Time-based split (train early, validate later)
# - Trains AE with EarlyStopping
# - Plots learning curve (reconstruction loss)
# - Scores train/valid, calibrates threshold, exports top-N anomalies

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, Model

# bring your featureizer
from bank_features_training_xls import build_training_matrix_from_excel  # must be in PYTHONPATH

# =========================
# CONFIG (edit these)
# =========================
HISTORY_PATH = "bank_history_3yrs.xlsx"   # path to your 3-year Excel
OUT_DIR      = "ae_outputs_keras_plain"
SHEET_NAME   = 0
ONE_HOT      = True        # must match how you want the featureizer to encode (True=onehot, False=ordinal)
CUTOFF       = None        # e.g., "2025-06-30" for fixed split; else None to use VALID_FRAC
VALID_FRAC   = 0.15        # used only if CUTOFF is None
BURNIN_TXN30D_MIN = 0      # drop early-history rows: keep rows with txn_count_30d >= this in TRAIN

SIZES        = (128, 64, 32)  # encoder: two hidden + bottleneck; mirrored decoder
L2           = 1e-6
DROPOUT      = 0.0
LR           = 1e-3
BATCH_SIZE   = 512
EPOCHS       = 200
PATIENCE     = 15
THRESHOLD_PERCENTILE = 99.0
TOPN_PER_GROUP = 10        # how many top anomalies to export per group (account/code)

# =========================
# Utilities
# =========================
def time_based_split(feats_df: pd.DataFrame, ts_col="ts", cutoff=None, valid_frac=0.15):
    df = feats_df.sort_values(ts_col).reset_index(drop=True)
    if cutoff:
        cut = pd.to_datetime(cutoff)
        train_mask = df[ts_col] < cut
        valid_mask = ~train_mask
    else:
        n = len(df)
        n_train = int(np.floor((1 - valid_frac) * n))
        train_mask = pd.Series([True] * n_train + [False] * (n - n_train))
        valid_mask = ~train_mask
    return train_mask.values, valid_mask.values

def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return ((x_true - x_pred) ** 2).mean(axis=1)

def make_autoencoder(input_dim: int, sizes=(128, 64, 32), l2=1e-6, dropout=0.0, lr=1e-3) -> Model:
    inp = layers.Input(shape=(input_dim,), name="in")
    x = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    bottleneck = layers.Dense(sizes[2], activation="relu", kernel_regularizer=regularizers.l2(l2), name="bottleneck")(x)
    x = layers.Dense(sizes[1], activation="relu", kernel_regularizer=regularizers.l2(l2))(bottleneck)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(sizes[0], activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    out = layers.Dense(input_dim, activation=None, name="recon")(x)
    ae = Model(inputs=inp, outputs=out, name="dense_autoencoder")
    ae.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=losses.MeanSquaredError())
    return ae

def train_autoencoder(ae: Model, X_train: np.ndarray, X_valid: np.ndarray, batch_size=512, epochs=200, patience=15):
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)
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

def plot_learning_curve(hist, out_png="learning_curve.png", title="Autoencoder Reconstruction Loss"):
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (reconstruction)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png

def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0):
    return float(np.percentile(valid_errors, percentile))

def topn_anomalies(df_scored: pd.DataFrame, score_col="recon_error", group_col=None, n=10):
    """
    If group_col is provided, returns top-n per group; else returns global top-n.
    """
    if group_col and group_col in df_scored.columns:
        parts = []
        for g, sub in df_scored.groupby(group_col):
            parts.append(sub.nlargest(n, score_col))
        return pd.concat(parts).sort_values(score_col, ascending=False)
    return df_scored.nlargest(n, score_col)

# =========================
# Orchestration
# =========================
def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Build matrix using the same transforms as inference
    X_all, feats_all, feat_names = build_training_matrix_from_excel(
        HISTORY_PATH, sheet_name=SHEET_NAME, one_hot=ONE_HOT
    )
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' timestamp in engineered features.")

    # 2) Time split
    train_mask, valid_mask = time_based_split(feats_all, ts_col="ts", cutoff=CUTOFF, valid_frac=VALID_FRAC)
    X_train, X_valid = X_all[train_mask], X_all[valid_mask]
    feats_train = feats_all.loc[train_mask].reset_index(drop=True)
    feats_valid = feats_all.loc[valid_mask].reset_index(drop=True)

    # Optional burn-in trimming
    if BURNIN_TXN30D_MIN > 0 and "txn_count_30d" in feats_train.columns:
        keep = feats_train["txn_count_30d"].values >= BURNIN_TXN30D_MIN
        X_train = X_train[keep]
        feats_train = feats_train.loc[keep].reset_index(drop=True)

    # 3) Model + training
    input_dim = X_all.shape[1]
    ae = make_autoencoder(input_dim, sizes=SIZES, l2=L2, dropout=DROPOUT, lr=LR)
    hist, ae = train_autoencoder(ae, X_train, X_valid, batch_size=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE)

    # 4) Learning curve
    curve_path = os.path.join(OUT_DIR, "learning_curve.png")
    plot_learning_curve(hist, out_png=curve_path)

    # 5) Scoring
    pred_train = ae.predict(X_train, batch_size=2048, verbose=0)
    pred_valid = ae.predict(X_valid, batch_size=2048, verbose=0)
    train_err = reconstruction_errors(X_train, pred_train)
    valid_err = reconstruction_errors(X_valid, pred_valid)

    # 6) Threshold
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # 7) Persist outputs
    ae.save(os.path.join(OUT_DIR, "ae_model_keras.h5"))
    meta = dict(
        input_dim=int(input_dim),
        sizes=list(SIZES),
        l2=float(L2),
        dropout=float(DROPOUT),
        lr=float(LR),
        epochs_ran=int(len(hist.history.get("loss", []))),
        val_loss=float(hist.history.get("val_loss", [np.nan])[-1]),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        threshold=float(thr),
        feature_names=feat_names,
        one_hot=bool(ONE_HOT),
        cutoff=CUTOFF,
        valid_frac=float(VALID_FRAC)
    )
    with open(os.path.join(OUT_DIR, "ae_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame({"recon_error": train_err}).to_csv(os.path.join(OUT_DIR, "train_scores.csv"), index=False)
    pd.DataFrame({"recon_error": valid_err}).to_csv(os.path.join(OUT_DIR, "valid_scores.csv"), index=False)

    feats_train_out = feats_train.copy()
    feats_train_out["recon_error"] = train_err
    feats_train_out.to_csv(os.path.join(OUT_DIR, "train_engineered_with_scores.csv"), index=False)

    feats_valid_out = feats_valid.copy()
    feats_valid_out["recon_error"] = valid_err
    feats_valid_out.to_csv(os.path.join(OUT_DIR, "valid_engineered_with_scores.csv"), index=False)

    # 8) Export top-N anomalies
    top_global = topn_anomalies(feats_valid_out, score_col="recon_error", group_col=None, n=TOPN_PER_GROUP)
    top_global.to_csv(os.path.join(OUT_DIR, "valid_topn_global.csv"), index=False)

    # per-account & per-code (if present)
    if "BankAccountCode" in feats_valid_out.columns:
        top_per_acct = topn_anomalies(feats_valid_out, score_col="recon_error",
                                      group_col="BankAccountCode", n=TOPN_PER_GROUP)
        top_per_acct.to_csv(os.path.join(OUT_DIR, "valid_topn_per_account.csv"), index=False)
    if "BankTransactionCode" in feats_valid_out.columns:
        top_per_code = topn_anomalies(feats_valid_out, score_col="recon_error",
                                      group_col="BankTransactionCode", n=TOPN_PER_GROUP)
        top_per_code.to_csv(os.path.join(OUT_DIR, "valid_topn_per_txncode.csv"), index=False)

    print(f"[done] dim={input_dim}  final_val_loss={hist.history.get('val_loss', [np.nan])[-1]:.6f}  "
          f"thr@p{THRESHOLD_PERCENTILE}={thr:.6f}")
    print(f"Outputs saved in: {OUT_DIR}")
    print(f"Learning curve: {curve_path}")

# =========================
# Kick off
# =========================
if __name__ == "__main__":
    run_pipeline()
