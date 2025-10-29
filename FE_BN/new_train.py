# ae_keras_fit_simple.py
# Simple Keras Autoencoder pipeline (no argparse, no burn-in).
# - Builds features via bank_features_training_xls.build_training_matrix_from_excel
# - Time-based split (train early, validate later)
# - Trains AE with EarlyStopping
# - Plots learning curve (MSE reconstruction)
# - Saves scores and engineered rows with scores

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, Model

from bank_features_training_xls import build_training_matrix_from_excel  # ensure import path

# =========================
# CONFIG — edit paths/sizes
# =========================
HISTORY_PATH = "bank_history_3yrs.xlsx"   # your 3-year Excel
OUT_DIR      = "ae_outputs_simple"
SHEET_NAME   = 0
ONE_HOT      = True        # must match featureizer mode you want (True=one-hot, False=ordinal)

# Split: choose either CUTOFF (fixed date) or VALID_FRAC (most recent fraction)
CUTOFF       = None        # e.g., "2025-06-30"; if None, uses VALID_FRAC
VALID_FRAC   = 0.15

# Model/training
SIZES        = (128, 64, 32)  # encoder sizes: two hidden + bottleneck, decoder mirrors
L2           = 1e-6
DROPOUT      = 0.0
LR           = 1e-3
BATCH_SIZE   = 512
EPOCHS       = 200
PATIENCE     = 15
THRESHOLD_PERCENTILE = 99.0

# =========================
# Helpers
# =========================
def time_based_split_indices(feats_df: pd.DataFrame,
                             ts_col: str = "ts",
                             cutoff: str | None = None,
                             valid_frac: float = 0.15):
    """
    Return integer indices (train_idx, valid_idx) w.r.t. ORIGINAL feats_df order.
    Uses timestamp column to decide ordering; never returns boolean masks.
    """
    ts = pd.to_datetime(feats_df[ts_col]).to_numpy()
    if cutoff:
        cut = np.datetime64(pd.to_datetime(cutoff))
        train_idx = np.flatnonzero(ts < cut)
        valid_idx = np.flatnonzero(ts >= cut)
        return train_idx, valid_idx
    order = np.argsort(ts)                       # earliest -> latest
    n = len(ts)
    n_train = int(np.floor((1.0 - valid_frac) * n))
    train_idx = order[:n_train]
    valid_idx = order[n_train:]
    return train_idx, valid_idx

def reconstruction_errors(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """Per-sample MSE reconstruction error."""
    return ((x_true - x_pred) ** 2).mean(axis=1)

def make_autoencoder(input_dim: int,
                     sizes=(128, 64, 32),
                     l2=1e-6,
                     dropout=0.0,
                     lr=1e-3) -> Model:
    """Function-only Keras AE (dense) using functional API."""
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

def train_autoencoder(ae: Model,
                      X_train: np.ndarray,
                      X_valid: np.ndarray,
                      batch_size=512,
                      epochs=200,
                      patience=15):
    """Train AE with EarlyStopping on val_loss; returns (history, ae_with_best_weights)."""
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

def plot_learning_curve(hist, out_png):
    """Plot train/valid reconstruction loss (MSE) vs epochs."""
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="valid loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE (reconstruction)")
    plt.title("Autoencoder Reconstruction Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def pick_threshold_from_validation(valid_errors: np.ndarray, percentile=99.0) -> float:
    return float(np.percentile(valid_errors, percentile))

# =========================
# Pipeline
# =========================
def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Build full design matrix from 3-year history (consistent with inference path)
    X_all, feats_all, feat_names = build_training_matrix_from_excel(
        HISTORY_PATH, sheet_name=SHEET_NAME, one_hot=ONE_HOT
    )
    assert X_all.shape[0] == len(feats_all), "Featureizer returned misaligned X_all vs feats_all."
    if "ts" not in feats_all.columns:
        raise ValueError("Expected 'ts' timestamp in engineered features.")

    # 2) Time-based split (index-based, avoids boolean-mask issues)
    train_idx, valid_idx = time_based_split_indices(feats_all, ts_col="ts",
                                                    cutoff=CUTOFF, valid_frac=VALID_FRAC)
    # Slice ONLY the full arrays with those indices
    X_train  = np.take(X_all, train_idx, axis=0)
    X_valid  = np.take(X_all, valid_idx, axis=0)
    feats_train = feats_all.iloc[train_idx].reset_index(drop=True)
    feats_valid = feats_all.iloc[valid_idx].reset_index(drop=True)

    # 3) Model + training
    input_dim = X_all.shape[1]
    ae = make_autoencoder(input_dim, sizes=SIZES, l2=L2, dropout=DROPOUT, lr=LR)
    hist, ae = train_autoencoder(ae, X_train, X_valid, batch_size=BATCH_SIZE, epochs=EPOCHS, patience=PATIENCE)

    # 4) Learning curve
    curve_path = os.path.join(OUT_DIR, "learning_curve.png")
    plot_learning_curve(hist, out_png=curve_path)

    # 5) Score reconstruction error
    pred_train = ae.predict(X_train, batch_size=2048, verbose=0)
    pred_valid = ae.predict(X_valid, batch_size=2048, verbose=0)
    train_err = reconstruction_errors(X_train, pred_train)
    valid_err = reconstruction_errors(X_valid, pred_valid)

    # 6) Threshold from validation
    thr = pick_threshold_from_validation(valid_err, percentile=THRESHOLD_PERCENTILE)

    # 7) Save outputs
    ae.save(os.path.join(OUT_DIR, "ae_model_keras.h5"))
    meta = dict(
        input_dim=int(input_dim),
        sizes=list(SIZES),
        l2=float(L2),
        dropout=float(DROPOUT),
        lr=float(LR),
        epochs_ran=int(len(hist.history.get("loss", []))),
        final_val_loss=float(hist.history.get("val_loss", [np.nan])[-1]),
        threshold_percentile=float(THRESHOLD_PERCENTILE),
        threshold=float(thr),
        feature_names=feat_names,
        one_hot=bool(ONE_HOT),
        cutoff=CUTOFF,
        valid_frac=float(VALID_FRAC),
        n_train=int(X_train.shape[0]),
        n_valid=int(X_valid.shape[0]),
    )
    with open(os.path.join(OUT_DIR, "ae_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame({"recon_error": train_err}).to_csv(os.path.join(OUT_DIR, "train_scores.csv"), index=False)
    pd.DataFrame({"recon_error": valid_err}).to_csv(os.path.join(OUT_DIR, "valid_scores.csv"), index=False)

    feats_train_out = feats_train.copy(); feats_train_out["recon_error"] = train_err
    feats_valid_out = feats_valid.copy(); feats_valid_out["recon_error"] = valid_err
    feats_train_out.to_csv(os.path.join(OUT_DIR, "train_engineered_with_scores.csv"), index=False)
    feats_valid_out.to_csv(os.path.join(OUT_DIR, "valid_engineered_with_scores.csv"), index=False)

    print(f"[done] dim={input_dim}  final_val_loss={meta['final_val_loss']:.6f}  "
          f"thr@p{THRESHOLD_PERCENTILE}={thr:.6f}")
    print(f"Outputs in: {OUT_DIR} | Learning curve: {curve_path}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    run_pipeline()
