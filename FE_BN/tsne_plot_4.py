# latent_tsne_per_combo.py
# t-SNE per (Account, BU, Code) combo using the trained combo AE.
# - Uses the same FE as training (bank_features_simple.build_dataset_from_df)
# - Loads combo_map.json + meta.json + combo_autoencoder.keras from OUT_DIR
# - For each combo that has anomalies:
#     * Compute bottleneck embeddings for train & test rows of that combo
#     * Also compute "model-normal" latent for each test row by encoding AE's reconstruction
#     * Run t-SNE on that combo only
#     * Plot:
#         - Train (green)
#         - Test normal (yellow)
#         - Test anomaly actual (red)
#         - Test anomaly predicted latent (blue star)
#         - 1σ and 2σ train cluster bands (ellipses in t-SNE space)

import os
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
from matplotlib.patches import Ellipse

# --- FE from your existing script ---
from bank_features_simple import build_dataset_from_df

# Should match training script
OUT_DIR_MODEL   = "combo_ae_outputs"
MODEL_PATH      = "combo_autoencoder.keras"
META_JSON       = "meta.json"
COMBO_MAP_JSON  = "combo_map.json"


# ========= Shared helpers (mirrors training script) =========

def signed_log1p(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


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
    stats: Dict[str, Dict[str, float]] = {}
    for combo, g in df_train.groupby("combo_str", sort=False):
        a = g["amount"].astype(float).values
        l = signed_log1p(a)
        med_log, iqr_log = median_iqr(l)
        mad = mad_inr(a)
        stats[combo] = {
            "median_log": med_log,
            "iqr_log": iqr_log,
            "mad_inr": mad,
            "count": int(len(a)),
        }
    return stats


def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    med_all, iqr_all = median_iqr(l)
    combo_list = df["combo_str"].astype(str).values
    med = np.array(
        [stats.get(c, {}).get("median_log", med_all) for c in combo_list],
        dtype=np.float32,
    )
    iqr = np.array(
        [stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list],
        dtype=np.float32,
    )
    iqr = np.maximum(iqr, 1e-6)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)


def build_inputs_with_ynorm(
    df: pd.DataFrame,
    tab_cols: List[str],
    y_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tab = (
        df[tab_cols].astype("float32").values
        if tab_cols
        else np.zeros((len(df), 0), dtype="float32")
    )
    y = y_norm.reshape(-1, 1).astype("float32")
    X_in = np.hstack([tab, y]).astype("float32")
    combo_ids = df["combo_id"].astype("int32").values
    return X_in, combo_ids


def apply_loaded_combo_map(df: pd.DataFrame, combo_map: Dict[str, int]) -> pd.DataFrame:
    d = df.copy()
    oov_id = combo_map.get("__OOV__", max(combo_map.values()))
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype(int)
    return d


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sanitize_for_filename(s: str) -> str:
    s = str(s)
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s)[:80]


# ========= Core preparation: latent vectors + anomaly mask =========

def prepare_latent_and_masks(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    anomalies_csv_path: str,
    out_dir_model: str = OUT_DIR_MODEL,
):
    """
    Rebuild FE, load model, compute:
      - bottleneck embeddings for TRAIN & TEST (Z_train, Z_test)
      - "model-normal" bottleneck embeddings for TEST (Z_test_pred) using AE.predict output
      - anomaly mask for test rows (anomaly_mask_test)

    Returns:
      feats_train, feats_test, Z_train, Z_test, Z_test_pred, anomaly_mask_test
    """
    # 1) Load artifacts
    meta_path = os.path.join(out_dir_model, META_JSON)
    combo_map_path = os.path.join(out_dir_model, COMBO_MAP_JSON)
    model_path = os.path.join(out_dir_model, MODEL_PATH)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    tab_cols = meta["tabular_feature_cols"]

    with open(combo_map_path, "r") as f:
        combo_map = json.load(f)

    # full AE model
    ae = tf.keras.models.load_model(model_path, compile=False)
    # encoder that outputs bottleneck from [x_in, combo_id]
    encoder = Model(inputs=ae.inputs, outputs=ae.get_layer("bottleneck").output)

    # 2) FE (same as training, from raw DFs)
    feats_train, _, _ = build_dataset_from_df(df_train_raw)
    feats_test, _, _ = build_dataset_from_df(df_test_raw)

    # Align tabular columns to what model actually used
    tab_cols = [
        c for c in tab_cols
        if c in feats_train.columns and c in feats_test.columns
    ]

    # 3) Apply combo map (train-time mapping + OOV)
    feats_train = apply_loaded_combo_map(feats_train, combo_map)
    feats_test = apply_loaded_combo_map(feats_test, combo_map)

    # 4) Stats + y_norm (TRAIN ONLY stats)
    stats_train = build_combo_stats_train(feats_train)
    y_train = normalize_amount(feats_train, stats_train)
    y_test = normalize_amount(feats_test, stats_train)

    # 5) AE inputs
    Xtr, Ctr = build_inputs_with_ynorm(feats_train, tab_cols, y_train)
    Xte, Cte = build_inputs_with_ynorm(feats_test, tab_cols, y_test)

    # 6) Latent/bottleneck vectors for actual inputs
    Z_train = encoder.predict([Xtr, Ctr], batch_size=2048, verbose=0)
    Z_test = encoder.predict([Xte, Cte], batch_size=2048, verbose=0)

    # 7) "Model-normal" latent for test points:
    #    - First get reconstructed input from AE.predict (this is exactly what is
    #      used for reconstruction error in training script)
    #    - Then encode that reconstructed input back through the encoder.
    Xte_pred = ae.predict([Xte, Cte], batch_size=2048, verbose=0)
    Z_test_pred = encoder.predict([Xte_pred, Cte], batch_size=2048, verbose=0)

    # 8) Anomaly mask for test rows
    anoms = pd.read_csv(anomalies_csv_path)

    # normalise keys (ts, combo_str, amount)
    ft = feats_test.copy()
    ft["ts"] = pd.to_datetime(ft["ts"])
    anoms["ts"] = pd.to_datetime(anoms["ts"])

    # ensure combo_str exists in anomalies
    if "combo_str" not in anoms.columns and {
        "BankAccountCode",
        "BusinessUnitCode",
        "BankTransactionCode",
    } <= set(anoms.columns):
        anoms["combo_str"] = (
            anoms["BankAccountCode"].astype(str)
            + "|"
            + anoms["BusinessUnitCode"].astype(str)
            + "|"
            + anoms["BankTransactionCode"].astype(str)
        )

    ft["amount"] = ft["amount"].astype(float)
    anoms["amount"] = anoms["amount"].astype(float)

    ft["__key"] = (
        ft["ts"].dt.strftime("%Y-%m-%d")
        + "|"
        + ft["combo_str"].astype(str)
        + "|"
        + ft["amount"].round(2).astype(str)
    )
    anoms["__key"] = (
        anoms["ts"].dt.strftime("%Y-%m-%d")
        + "|"
        + anoms["combo_str"].astype(str)
        + "|"
        + anoms["amount"].round(2).astype(str)
    )

    anom_keys = set(anoms["__key"].tolist())
    anomaly_mask_test = ft["__key"].isin(anom_keys).values

    return feats_train, feats_test, Z_train, Z_test, Z_test_pred, anomaly_mask_test


# ========= t-SNE per combo =========

def _plot_sigma_ellipses(ax, pts: np.ndarray, n_std_list=(1.0, 2.0)) -> None:
    """
    Draw 1σ and 2σ ellipses for the training cluster in t-SNE space.
    pts: (n,2) array of train points in t-SNE coordinates.
    """
    if pts.shape[0] < 3:
        return

    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)

    # eigen decomposition for ellipse axes
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort largest first
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    for n_std, alpha in zip(n_std_list, [0.08, 0.04]):
        width, height = 2 * n_std * np.sqrt(eigvals)
        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="grey",
            facecolor="grey",
            alpha=alpha,
            linestyle="--",
            linewidth=1.0,
            label=f"{int(n_std)}σ band" if n_std == 1.0 else f"{int(n_std)}σ band",
        )
        ax.add_patch(ell)


def plot_tsne_for_single_combo(
    combo_str: str,
    feats_train: pd.DataFrame,
    feats_test: pd.DataFrame,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    Z_test_pred: np.ndarray,
    anomaly_mask_test: np.ndarray,
    out_dir_plots: str,
    show: bool = True,
    min_points: int = 5,
) -> str:
    """
    Run t-SNE only for one combo.
    Train = green, Test normal = yellow, Test anomaly actual = red,
    Test anomaly predicted latent = blue star.
    Also draws 1σ and 2σ ellipses for the TRAIN cluster in t-SNE space.

    Returns path to saved PNG (or '' if skipped).
    """
    mask_tr = feats_train["combo_str"].astype(str) == combo_str
    mask_te = feats_test["combo_str"].astype(str) == combo_str

    n_tr = mask_tr.sum()
    n_te = mask_te.sum()
    if n_tr + n_te < min_points:
        print(f"[SKIP] Combo '{combo_str}' has only {n_tr + n_te} points.")
        return ""

    Z_tr_combo = Z_train[mask_tr]
    Z_te_combo = Z_test[mask_te]
    Z_te_pred_combo = Z_test_pred[mask_te]

    # anomaly vs normal within this combo's test points
    anom_combo_mask = anomaly_mask_test[mask_te]
    n_anom = anom_combo_mask.sum()
    if n_anom == 0:
        print(f"[INFO] Combo '{combo_str}' has no anomalies; skipping.")
        return ""

    # Stack all (train + test actual + test predicted) for a single t-SNE fit
    Z_all = np.vstack([Z_tr_combo, Z_te_combo, Z_te_pred_combo])
    n_total = Z_all.shape[0]

    # t-SNE params adapted to sample size
    perplexity = max(5, min(30, n_total // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    Z_2d = tsne.fit_transform(Z_all)

    n_pred = Z_te_pred_combo.shape[0]
    Z_tr_2d = Z_2d[:n_tr]
    Z_te_2d = Z_2d[n_tr : n_tr + n_te]
    Z_te_pred_2d = Z_2d[n_tr + n_te : n_tr + n_te + n_pred]

    # split test into normal vs anomaly (for both actual and predicted)
    Z_te_anom_2d = Z_te_2d[anom_combo_mask]
    Z_te_norm_2d = Z_te_2d[~anom_combo_mask]

    Z_te_pred_anom_2d = Z_te_pred_2d[anom_combo_mask]
    # (if you ever want predicted-normal as well, you can similarly index ~anom_combo_mask)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Train cluster with 1σ / 2σ bands
    _plot_sigma_ellipses(ax, Z_tr_2d, n_std_list=(1.0, 2.0))

    # Train: green
    ax.scatter(
        Z_tr_2d[:, 0],
        Z_tr_2d[:, 1],
        s=25,
        c="green",
        alpha=0.6,
        label="Train",
    )
    # Test normal: yellow
    if len(Z_te_norm_2d) > 0:
        ax.scatter(
            Z_te_norm_2d[:, 0],
            Z_te_norm_2d[:, 1],
            s=35,
            c="gold",
            edgecolors="black",
            alpha=0.9,
            label="Test normal",
        )
    # Test anomaly actual: red
    ax.scatter(
        Z_te_anom_2d[:, 0],
        Z_te_anom_2d[:, 1],
        s=60,
        c="red",
        edgecolors="black",
        alpha=0.9,
        label="Test anomaly (actual)",
    )
    # Test anomaly predicted latent: blue star
    ax.scatter(
        Z_te_pred_anom_2d[:, 0],
        Z_te_pred_anom_2d[:, 1],
        s=120,
        c="blue",
        marker="*",
        edgecolors="black",
        alpha=1.0,
        label="Test anomaly (model-normal)",
    )

    acct, bu, code = combo_str.split("|")
    ax.set_title(f"t-SNE (combo) – Acct={acct}, BU={bu}, Code={code}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    safe = _sanitize_for_filename(combo_str)
    fname = f"latent_tsne_combo_{safe}.png"
    _mkdir(out_dir_plots)
    out_path = os.path.join(out_dir_plots, fname)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[SAVED] {out_path}")
    return out_path


def plot_tsne_per_combo_for_anomalies(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    anomalies_csv_path: str,
    out_dir_model: str = OUT_DIR_MODEL,
    out_dir_plots: str = "latent_tsne_per_combo",
    max_combos: int = 20,
    show: bool = True,
) -> List[str]:
    """
    High-level helper:
      - Prepares latent representations + "model-normal" latent + anomaly mask.
      - Finds combos that have anomalies.
      - For each such combo (up to max_combos), builds a per-combo t-SNE chart.

    Returns list of saved PNG paths.
    """
    (
        feats_train,
        feats_test,
        Z_train,
        Z_test,
        Z_test_pred,
        anomaly_mask_test,
    ) = prepare_latent_and_masks(
        df_train_raw=df_train_raw,
        df_test_raw=df_test_raw,
        anomalies_csv_path=anomalies_csv_path,
        out_dir_model=out_dir_model,
    )

    anoms = pd.read_csv(anomalies_csv_path)
    if "combo_str" not in anoms.columns and {
        "BankAccountCode",
        "BusinessUnitCode",
        "BankTransactionCode",
    } <= set(anoms.columns):
        anoms["combo_str"] = (
            anoms["BankAccountCode"].astype(str)
            + "|"
            + anoms["BusinessUnitCode"].astype(str)
            + "|"
            + anoms["BankTransactionCode"].astype(str)
        )

    combos = anoms["combo_str"].astype(str).unique().tolist()
    print(f"[INFO] Found {len(combos)} combos with anomalies; plotting up to {max_combos}.")

    paths = []
    for combo_str in combos[:max_combos]:
        p = plot_tsne_for_single_combo(
            combo_str=combo_str,
            feats_train=feats_train,
            feats_test=feats_test,
            Z_train=Z_train,
            Z_test=Z_test,
            Z_test_pred=Z_test_pred,
            anomaly_mask_test=anomaly_mask_test,
            out_dir_plots=out_dir_plots,
            show=show,
        )
        if p:
            paths.append(p)
    return paths
