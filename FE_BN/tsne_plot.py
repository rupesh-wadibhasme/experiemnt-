# latent_tsne.py
# t-SNE visualisation of combo autoencoder latent space
#
# Usage from notebook (external test setup):
#
#   import pandas as pd
#   from latent_tsne import plot_latent_tsne
#
#   df_train_all = pd.read_excel("year1_full.xlsx", sheet_name=0)
#   df_test_all  = pd.read_excel("df_year2_jan_feb.xlsx", sheet_name=0)
#
#   fig, tsne_df = plot_latent_tsne(
#       df_train_all=df_train_all,
#       df_test_all=df_test_all,
#       anoms_csv_path="combo_ae_outputs/anomalies_combo_ae.csv",
#       model_path="combo_ae_outputs/combo_autoencoder.keras",
#       out_png="combo_ae_outputs/latent_tsne.png",
#       n_train_sample=5000,
#       perplexity=30,
#       random_state=42,
#       show=True,
#   )

import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf


import os
import json


COMBO_MAP_NAME = "combo_map.json"

def apply_combo_map_for_tsne(df: pd.DataFrame, combo_map: dict) -> pd.DataFrame:
    """
    Use the SAME combo_map that was used during training.
    Unseen combos are mapped to the OOV id.
    """
    d = df.copy()
    oov_id = combo_map[OOV_ID_NAME] if OOV_ID_NAME in combo_map else max(combo_map.values())
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype("int32")
    return d


# import helpers from your AE training script
from training_combo_autoencoder_df import (
    build_dataset_from_df,
    split_train_valid_tail_full,
    make_combo_map_from_train,
    apply_combo_map,
    build_combo_stats_train,
    normalize_amount,
    build_inputs_with_ynorm,
)

# -----------------------------
# Build AE inputs for t-SNE
# -----------------------------
def build_ae_inputs_for_tsne(
    df_train_all: pd.DataFrame,
    df_test_all: pd.DataFrame,
    valid_frac: float = 0.10,
) -> Tuple[
    pd.DataFrame, pd.DataFrame,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Dict[str, float], Dict[str, int], int, int
]:
    """
    Rebuild AE inputs exactly as in external-test training pipeline,
    but returning full TRAIN (train+valid) and TEST matrices.

    Returns:
      df_train_full  : engineered TRAIN+VALID DataFrame (with combo_id etc.)
      df_test        : engineered TEST DataFrame (with combo_id etc.)
      X_train        : AE numeric input for train (tabular + y_norm)
      C_train        : combo_id vector for train
      X_test         : AE numeric input for test  (tabular + y_norm)
      C_test         : combo_id vector for test
      stats_train    : per-combo stats dict (median_log, iqr_log, mad_inr, count)
      combo_map      : dict combo_str -> combo_id (with OOV)
      j_y            : column index of y_norm in X_* (last column)
      n_tab          : number of tabular feature columns (X_* shape[1] - 1)
    """
    # --- FE on TRAIN df ---
    feats_train_all, tab_cols_train, _ = build_dataset_from_df(df_train_all)

    # split train into train-core and valid-tail (time-wise)
    df_train_core, df_valid = split_train_valid_tail_full(
        feats_train_all, ts_col="ts", valid_frac=valid_frac
    )

    # build combo map from TRAIN core only (same as training)
    combo_map = make_combo_map_from_train(df_train_core)
    df_train_core = apply_combo_map(df_train_core, combo_map)
    df_valid = apply_combo_map(df_valid, combo_map)

    # --- FE on TEST df ---
    feats_test_all, tab_cols_test, _ = build_dataset_from_df(df_test_all)

    # align tabular columns (use TRAIN's tab_cols)
    tab_cols = [c for c in tab_cols_train if c in feats_test_all.columns]
    df_test = apply_combo_map(feats_test_all, combo_map)

    # full TRAIN = train-core + valid (for t-SNE, both are "normal" training)
    df_train_full = pd.concat([df_train_core, df_valid], ignore_index=True)

    # --- per-combo stats from TRAIN core only ---
    stats_train = build_combo_stats_train(df_train_core)

    # --- y_norm for train_full & test (using TRAIN stats only) ---
    y_train_full = normalize_amount(df_train_full, stats_train)
    y_test = normalize_amount(df_test, stats_train)

    # --- AE inputs (tabular + y_norm) ---
    X_train, C_train, col_names, j_y = build_inputs_with_ynorm(
        df_train_full, tab_cols, y_train_full
    )
    X_test, C_test, _, _ = build_inputs_with_ynorm(
        df_test, tab_cols, y_test
    )

    n_tab = len(tab_cols)

    return (
        df_train_full,
        df_test,
        X_train,
        C_train,
        X_test,
        C_test,
        stats_train,
        combo_map,
        j_y,
        n_tab,
    )


# -----------------------------
# Load encoder (bottleneck) from trained AE
# -----------------------------
def load_encoder(model_path: str) -> tf.keras.Model:
    """
    Load the trained combo autoencoder and expose only the bottleneck layer.
    Assumes the layer is named 'bottleneck' (as in training script).
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    if "bottleneck" not in [l.name for l in model.layers]:
        raise ValueError("Model does not contain a layer named 'bottleneck'.")
    encoder = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("bottleneck").output
    )
    return encoder


# -----------------------------
# Compute latent representations
# -----------------------------
def compute_latent(
    encoder: tf.keras.Model,
    X_train: np.ndarray,
    C_train: np.ndarray,
    X_test: np.ndarray,
    C_test: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the encoder to obtain latent vectors for train and test.
    """
    z_train = encoder.predict(
        [X_train, C_train], batch_size=batch_size, verbose=0
    )
    z_test = encoder.predict(
        [X_test, C_test], batch_size=batch_size, verbose=0
    )
    return z_train, z_test


# -----------------------------
# Mark anomalies in TEST via anomalies CSV
# -----------------------------
def mark_anomalies_with_scores(
    df_test: pd.DataFrame,
    anoms_csv_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align anomalies from CSV to df_test using (ts, combo_str, amount) keys.
    Returns:
      is_anomaly: boolean array aligned to df_test.index
      recon_error: float array (NaN for non-anomalies if not present)
    """
    anoms = pd.read_csv(anoms_csv_path)

    # key columns expected in anomalies CSV
    required_cols = ["ts", "combo_str", "amount"]
    for c in required_cols:
        if c not in anoms.columns:
            raise KeyError(f"Anomalies CSV missing required column '{c}'.")

    # convert keys
    df_t = df_test.copy()
    df_t["ts_key"] = pd.to_datetime(df_t["ts"])
    anoms["ts_key"] = pd.to_datetime(anoms["ts"])

    df_t["amount_key"] = df_t["amount"].astype(float).round(6)
    anoms["amount_key"] = anoms["amount"].astype(float).round(6)

    # subset anomalies columns to merge
    cols_to_merge = ["ts_key", "combo_str", "amount_key"]
    if "recon_error" in anoms.columns:
        cols_to_merge.append("recon_error")

    merged = df_t.merge(
        anoms[cols_to_merge],
        on=["ts_key", "combo_str", "amount_key"],
        how="left",
        indicator="an_flag",
    )

    is_anom = merged["an_flag"].eq("both").values

    if "recon_error" in anoms.columns:
        recon = merged["recon_error"].astype(float).values
        recon[~is_anom] = np.nan
    else:
        recon = np.full(len(df_test), np.nan, dtype=float)

    return is_anom, recon


# -----------------------------
# Run t-SNE on latent space
# -----------------------------
def run_tsne_latent(
    z_train: np.ndarray,
    z_test: np.ndarray,
    n_train_sample: int = 5000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run t-SNE on a subset of train latent vectors plus all test latent vectors.
    Returns:
      coords_train: (n_train_sub, 2)
      coords_test:  (n_test, 2)
      idx_train_sub: indices of the sampled train points (into original z_train)
    """
    n_tr = z_train.shape[0]
    rng = np.random.RandomState(random_state)

    if n_tr > n_train_sample:
        idx_train_sub = rng.choice(n_tr, size=n_train_sample, replace=False)
        z_train_sub = z_train[idx_train_sub]
    else:
        idx_train_sub = np.arange(n_tr)
        z_train_sub = z_train

    z_all = np.vstack([z_train_sub, z_test])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    coords_all = tsne.fit_transform(z_all)

    coords_train = coords_all[: len(z_train_sub)]
    coords_test = coords_all[len(z_train_sub) :]

    return coords_train, coords_test, idx_train_sub


# -----------------------------
# Main plotting function
# -----------------------------
def plot_latent_tsne(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    model_path: str,
    meta_path: str,
    anomalies_csv: str,
    out_png: str = "latent_tsne.png",
    show: bool = False,
    perplexity: float = 30.0,
    random_state: int = 42,
):
    """
    Build a t-SNE plot in bottleneck (latent) space:
      - Blue: train points
      - Grey: test normals
      - Red: test anomalies

    Uses the SAME combo_map as training (combo_map.json), so embedding indices stay valid.
    """

    # --- 1) Load meta & combo map (from training output dir) ---
    model_dir = os.path.dirname(meta_path) if meta_path else os.path.dirname(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tab_cols = meta["tabular_feature_cols"]

    combo_map_path = os.path.join(model_dir, COMBO_MAP_NAME)
    with open(combo_map_path, "r") as f:
        combo_map = json.load(f)

    # --- 2) Feature engineering on raw dataframes ---
    feats_train_all, _, _ = build_dataset_from_df(df_train_raw)
    feats_test_all,  _, _ = build_dataset_from_df(df_test_raw)

    # Apply SAME combo map (with OOV for unseen)
    feats_train = apply_combo_map_for_tsne(feats_train_all, combo_map)
    feats_test  = apply_combo_map_for_tsne(feats_test_all,  combo_map)

    # Align tabular cols using training meta
    tab_cols = [c for c in tab_cols if c in feats_train.columns and c in feats_test.columns]

    # --- 3) Train-only stats for y_norm ---
    stats_train = build_combo_stats_train(feats_train)

    y_train = normalize_amount(feats_train, stats_train)
    y_test  = normalize_amount(feats_test,  stats_train)

    # AE inputs
    Xtr, Ctr, col_names, j_y = build_inputs_with_ynorm(feats_train, tab_cols, y_train)
    Xte, Cte, _,        _    = build_inputs_with_ynorm(feats_test,  tab_cols, y_test)

    # --- 4) Load model & build encoder (bottleneck) ---
    model = tf.keras.models.load_model(model_path, compile=False)
    bottleneck_layer = model.get_layer("bottleneck")
    encoder = tf.keras.Model(inputs=model.inputs, outputs=bottleneck_layer.output)

    # This is where your error happened earlier:
    # now Ctr, Cte are guaranteed to be within [0, input_dim) of the embedding.
    Z_train = encoder.predict([Xtr, Ctr], batch_size=2048, verbose=0)
    Z_test  = encoder.predict([Xte, Cte], batch_size=2048, verbose=0)

    # --- 5) Load anomalies (on test set) to mark them in latent space ---
    anoms = pd.read_csv(anomalies_csv)
    anoms_key = (
        anoms["BankAccountCode"].astype(str) + "|" +
        anoms["BusinessUnitCode"].astype(str) + "|" +
        anoms["BankTransactionCode"].astype(str) + "|" +
        pd.to_datetime(anoms["ts"]).astype(str)
    )

    test_key = (
        feats_test["BankAccountCode"].astype(str) + "|" +
        feats_test["BusinessUnitCode"].astype(str) + "|" +
        feats_test["BankTransactionCode"].astype(str) + "|" +
        pd.to_datetime(feats_test["ts"]).astype(str)
    )

    is_anom_test = test_key.isin(anoms_key).values

    # --- 6) Build joint latent matrix for t-SNE ---
    Z_all = np.vstack([Z_train, Z_test])
    labels = np.concatenate([
        np.zeros(len(Z_train), dtype=int),                 # 0 = train
        np.where(is_anom_test, 2, 1).astype(int)           # 1 = test normal, 2 = test anomaly
    ])

    # --- 7) t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    Z_2d = tsne.fit_transform(Z_all)

    # --- 8) Plot ---
    plt.figure(figsize=(8, 6))

    mask_train = labels == 0
    mask_test_norm = labels == 1
    mask_test_anom = labels == 2

    plt.scatter(Z_2d[mask_train, 0], Z_2d[mask_train, 1], s=10, alpha=0.4, label="Train")
    plt.scatter(Z_2d[mask_test_norm, 0], Z_2d[mask_test_norm, 1], s=10, alpha=0.4, label="Test normal")
    plt.scatter(Z_2d[mask_test_anom, 0], Z_2d[mask_test_anom, 1], s=40, alpha=0.9, label="Test anomaly")

    plt.title("t-SNE of AE bottleneck (combo-aware)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    if show:
        plt.show()

    plt.close()

    print(f"[SAVED] {out_png}")
    return Z_2d, labels


from latent_tsne_combo import plot_latent_tsne

# df_year1, df_year2_janfeb are your raw dataframes
Z_2d, labels = plot_latent_tsne(
    df_train_raw=df_year1,
    df_test_raw=df_year2_janfeb,
    model_path="combo_ae_outputs/combo_autoencoder.keras",
    meta_path="combo_ae_outputs/meta.json",
    anomalies_csv="combo_ae_outputs/anomalies_combo_ae.csv",
    out_png="latent_tsne_combo.png",
    show=True,      # <- will display inline
)
