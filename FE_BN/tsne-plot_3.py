# latent_tsne_per_combo.py
# t-SNE per (Account, BU, Code) combo using the trained combo AE.
# - Uses the same FE as training (bank_features_simple.build_dataset_from_df)
# - Loads combo_map.json + meta.json + combo_autoencoder.keras from OUT_DIR
# - For each combo that has anomalies:
#     * Compute bottleneck embeddings for train & test rows of that combo
#     * Run t-SNE on that combo only
#     * Plot Train (green), Test normal (yellow), Test anomaly (red)

import os
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model

# --- FE from your existing script ---
from bank_features_simple import build_dataset_from_df

# Should match training script
OUT_DIR_MODEL = "combo_ae_outputs"
MODEL_PATH    = "combo_autoencoder.keras"
META_JSON     = "meta.json"
COMBO_MAP_JSON = "combo_map.json"


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
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 1e-6)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)


def build_inputs_with_ynorm(
    df: pd.DataFrame,
    tab_cols: List[str],
    y_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros((len(df), 0), dtype="float32")
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
    Rebuild FE, load model, compute bottleneck embeddings and anomaly mask for test.
    Returns:
      feats_train, feats_test, Z_train, Z_test, anomaly_mask_test
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

    ae = tf.keras.models.load_model(model_path, compile=False)
    encoder = Model(inputs=ae.inputs, outputs=ae.get_layer("bottleneck").output)

    # 2) FE (same as training, from raw DFs)
    feats_train, _, _ = build_dataset_from_df(df_train_raw)
    feats_test, _, _ = build_dataset_from_df(df_test_raw)

    # Align tabular columns to what model actually used
    tab_cols = [c for c in tab_cols if c in feats_train.columns and c in feats_test.columns]

    # 3) Apply combo map (train-time mapping + OOV)
    feats_train = apply_loaded_combo_map(feats_train, combo_map)
    feats_test = apply_loaded_combo_map(feats_test, combo_map)

    # 4) Stats + y_norm
    stats_train = build_combo_stats_train(feats_train)
    y_train = normalize_amount(feats_train, stats_train)
    y_test = normalize_amount(feats_test, stats_train)

    # 5) AE inputs
    Xtr, Ctr = build_inputs_with_ynorm(feats_train, tab_cols, y_train)
    Xte, Cte = build_inputs_with_ynorm(feats_test, tab_cols, y_test)

    # 6) Latent/bottleneck vectors
    Z_train = encoder.predict([Xtr, Ctr], batch_size=2048, verbose=0)
    Z_test = encoder.predict([Xte, Cte], batch_size=2048, verbose=0)

    # 7) Anomaly mask for test rows
    anoms = pd.read_csv(anomalies_csv_path)
    # normalise keys (ts, combo_str, amount)
    ft = feats_test.copy()
    ft["ts"] = pd.to_datetime(ft["ts"])
    anoms["ts"] = pd.to_datetime(anoms["ts"])

    # ensure required cols
    if "combo_str" not in anoms.columns and {"BankAccountCode", "BusinessUnitCode", "BankTransactionCode"} <= set(anoms.columns):
        anoms["combo_str"] = (
            anoms["BankAccountCode"].astype(str)
            + "|"
            + anoms["BusinessUnitCode"].astype(str)
            + "|"
            + anoms["BankTransactionCode"].astype(str)
        )

    # 'amount' should be there already from training script
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

    return feats_train, feats_test, Z_train, Z_test, anomaly_mask_test


# ========= t-SNE per combo =========


def _plot_sigma_ellipses(ax, pts_2d: np.ndarray, n_std_list=(1.0, 2.0)):
    """
    Draw Gaussian ellipses (1σ, 2σ, ...) for the cluster of 2D points.
    pts_2d: array of shape (n_samples, 2)
    """
    if pts_2d.shape[0] < 3:
        # not enough points to estimate covariance in a stable way
        return

    mean = pts_2d.mean(axis=0)
    cov = np.cov(pts_2d.T)

    # Eigen-decomposition of covariance (2x2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort by descending eigenvalue (largest axis first)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])  # (2, 200)

    # colors / linestyles for boundaries
    styles = {
        n_std_list[0]: {"linestyle": "-",  "linewidth": 1.0, "alpha": 0.7, "label": "±1 sigma band"},
        n_std_list[1]: {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7, "label": "±2 sigma band"},
    }

    for n_std in n_std_list:
        # semi-axis lengths for this sigma level
        axis_lengths = n_std * np.sqrt(np.maximum(eigvals, 1e-12))
        # transform circle -> ellipse
        ellipse = eigvecs @ np.diag(axis_lengths) @ circle  # (2,200)
        ellipse = ellipse.T + mean  # (200,2) + mean

        style = styles.get(n_std, {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7, "label": f"±{n_std} sigma"})
        ax.plot(ellipse[:, 0], ellipse[:, 1], **style)


def plot_tsne_for_single_combo(
    Z_train_combo: np.ndarray,   # bottleneck for train rows of this combo, shape (n_tr, latent_dim)
    Z_test_combo: np.ndarray,    # bottleneck for test rows of this combo, shape (n_te, latent_dim)
    is_anom_test: np.ndarray,    # boolean mask over test rows of this combo, shape (n_te,)
    combo_label: str,
    perplexity: float = 10.0,
    random_state: int = 42,
    out_path: str = None,
    show: bool = True,
):
    """
    Make a t-SNE plot for a single (Account, BU, Code) combo.

    - Train points    -> green
    - Test normal     -> yellow
    - Test anomalies  -> red
    - Overlays 1σ and 2σ ellipses around the TRAIN cluster in t-SNE space.

    Z_train_combo, Z_test_combo are bottleneck vectors from the AE.
    """

    # --- 1. Prepare data for t-SNE ---
    n_tr = Z_train_combo.shape[0]
    n_te = Z_test_combo.shape[0]

    if n_tr == 0 or n_te == 0:
        print(f"[WARN] Not enough data for combo {combo_label} (train={n_tr}, test={n_te})")
        return

    Z_all = np.vstack([Z_train_combo, Z_test_combo])

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, (Z_all.shape[0] - 1) / 3)),  # keep perplexity sane
        learning_rate="auto",
        init="random",
        random_state=random_state,
    )
    Z_all_2d = tsne.fit_transform(Z_all)

    Z_tr_2d = Z_all_2d[:n_tr]
    Z_te_2d = Z_all_2d[n_tr:]

    # --- 2. Split test into normal vs anomaly ---
    is_anom_test = np.asarray(is_anom_test, dtype=bool)
    if is_anom_test.shape[0] != n_te:
        raise ValueError(f"is_anom_test length {is_anom_test.shape[0]} != n_test {n_te}")

    Z_te_anom  = Z_te_2d[is_anom_test]
    Z_te_norm  = Z_te_2d[~is_anom_test]

    # --- 3. Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # Train cluster
    ax.scatter(
        Z_tr_2d[:, 0], Z_tr_2d[:, 1],
        s=30, alpha=0.7, label="Train", color="green"
    )

    # Test normal
    if Z_te_norm.shape[0] > 0:
        ax.scatter(
            Z_te_norm[:, 0], Z_te_norm[:, 1],
            s=40, alpha=0.9, label="Test normal", color="gold"
        )

    # Test anomalies
    if Z_te_anom.shape[0] > 0:
        ax.scatter(
            Z_te_anom[:, 0], Z_te_anom[:, 1],
            s=60, alpha=0.95, label="Test anomaly", color="red", edgecolors="k"
        )

    # --- 4. Add 1σ and 2σ cluster boundaries for TRAIN points ---
    _plot_sigma_ellipses(ax, Z_tr_2d, n_std_list=(1.0, 2.0))

    ax.set_title(f"t-SNE (combo) - {combo_label}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        print(f"[SAVED] {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


from latent_tsne_per_combo import plot_tsne_per_combo_for_anomalies

paths = plot_tsne_per_combo_for_anomalies(
    df_train_raw=df_train_year1,
    df_test_raw=df_test_jan_feb,
    anomalies_csv_path="combo_ae_outputs/anomalies_combo_ae.csv",
    out_dir_model="combo_ae_outputs",
    out_dir_plots="latent_tsne_per_combo",
    max_combos=10,   # top N combos with anomalies
    show=True        # show inline + save PNGs
)
paths
