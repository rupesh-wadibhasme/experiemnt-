# ae_anomaly_feature_explainer.py
# Explain AE anomalies via per-feature reconstruction-error contributions.
#
# Usage in notebook (example):
#   from ae_anomaly_feature_explainer import (
#       prepare_explainer_state,
#       explain_single_anomaly_by_index,
#       explain_all_anomalies_from_csv,
#   )
#
#   state = prepare_explainer_state(df_train_year1, df_test_jan_feb)
#
#   # A) Single anomaly (by test-row index)
#   explain_single_anomaly_by_index(
#       state,
#       test_index=123,
#       focus_features=["amount", "month", "cashbook_flag_derived"],
#       out_dir="ae_anomaly_explanations",
#       show=True,
#   )
#
#   # B) All anomalies from CSV (up to max_n)
#   explain_all_anomalies_from_csv(
#       state,
#       anomalies_csv_path="combo_ae_outputs/anomalies_combo_ae.csv",
#       focus_features=["amount", "month", "cashbook_flag_derived"],
#       max_anomalies=20,
#       out_dir="ae_anomaly_explanations",
#       show=False,
#   )

import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

from bank_features_simple import build_dataset_from_df

# Paths should match training script
OUT_DIR_MODEL   = "combo_ae_outputs"
MODEL_PATH      = "combo_autoencoder.keras"
META_JSON       = "meta.json"
COMBO_MAP_JSON  = "combo_map.json"


# ========= Helpers (mirroring training script) =========

def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def signed_log1p(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def inv_signed_log1p(lv: np.ndarray) -> np.ndarray:
    return np.sign(lv) * (np.expm1(np.abs(lv)))


def median_iqr(x: np.ndarray) -> Tuple[float, float]:
    q50 = float(np.median(x))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = max(q75 - q25, 1e-6)
    return q50, iqr


def build_combo_stats_train(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Per-combo stats from TRAIN ONLY:
      - median_log and iqr_log of signed_log1p(amount)
      - mad_inr is not needed here (only for anomaly gating), so omitted
    """
    stats: Dict[str, Dict[str, float]] = {}
    for combo, g in df_train.groupby("combo_str", sort=False):
        a = g["amount"].astype(float).values
        l = signed_log1p(a)
        med_log, iqr_log = median_iqr(l)
        stats[combo] = {
            "median_log": med_log,
            "iqr_log": iqr_log,
            "count": int(len(a)),
        }
    return stats


def normalize_amount(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    y_norm = (signed_log1p(amount) - median_log_combo) / iqr_log_combo
    """
    a = df["amount"].astype(float).values
    l = signed_log1p(a)

    med_all, iqr_all = median_iqr(l)
    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", med_all) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", iqr_all) for c in combo_list], dtype=np.float32)
    iqr = np.maximum(iqr, 1e-6)

    y_norm = (l - med) / iqr
    return y_norm.astype(np.float32)


def invert_pred_to_inr(df: pd.DataFrame,
                       y_norm_pred: np.ndarray,
                       stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    amount_pred_inr = inv_signed_log1p( y_norm_pred * iqr_log_combo + median_log_combo )
    """
    combo_list = df["combo_str"].astype(str).values
    med = np.array([stats.get(c, {}).get("median_log", 0.0) for c in combo_list], dtype=np.float32)
    iqr = np.array([stats.get(c, {}).get("iqr_log", 1.0)  for c in combo_list], dtype=np.float32)
    lv = y_norm_pred.reshape(-1) * iqr + med
    return inv_signed_log1p(lv)


def build_inputs_with_ynorm(df: pd.DataFrame,
                            tab_cols: List[str],
                            y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Returns:
      X_in: float32 matrix of [tabular_features, y_norm]  (this is what AE reconstructs)
      X_combo: int32 vector of combo_id                  (goes through embedding)
      col_names: names corresponding to X_in columns
      j_y: column index of y_norm within X_in
    """
    tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros((len(df), 0), dtype="float32")
    y = y_norm.reshape(-1, 1).astype("float32")
    X_in = np.hstack([tab, y]).astype("float32")
    X_combo = df["combo_id"].astype("int32").values
    col_names = list(tab_cols) + ["y_norm"]
    j_y = len(col_names) - 1
    return X_in, X_combo, col_names, j_y


def apply_loaded_combo_map(df: pd.DataFrame, combo_map: Dict[str, int]) -> pd.DataFrame:
    d = df.copy()
    # assume training script used "__OOV__" for unseen combos
    oov_id = combo_map.get("__OOV__", max(combo_map.values()))
    d["combo_id"] = d["combo_str"].map(combo_map).fillna(oov_id).astype(int)
    return d


def compute_row_recon_error(X_true: np.ndarray,
                            X_pred: np.ndarray,
                            col_w: Optional[np.ndarray]) -> np.ndarray:
    """
    Weighted MSE per row (matches training loss definition).
    """
    if col_w is None:
        se = (X_true - X_pred) ** 2
        return se.mean(axis=1)
    se = ((X_true - X_pred) ** 2) * col_w.reshape(1, -1)
    return se.mean(axis=1)


# ========= State preparation =========

def prepare_explainer_state(df_train_raw: pd.DataFrame,
                            df_test_raw: pd.DataFrame,
                            out_dir_model: str = OUT_DIR_MODEL) -> Dict[str, Any]:
    """
    Build all state needed for anomaly explanation:
      - FE on train & test
      - load model, meta, combo_map
      - compute y_norm, AE inputs, column weights

    Returns a dict 'state' with everything required by explain_* functions.
    """
    # 1) Load artifacts
    meta_path = os.path.join(out_dir_model, META_JSON)
    combo_map_path = os.path.join(out_dir_model, COMBO_MAP_JSON)
    model_path = os.path.join(out_dir_model, MODEL_PATH)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta_tab_cols = meta["tabular_feature_cols"]
    weight_y_norm = float(meta.get("weight_y_norm", 5.0))
    weight_tabular = float(meta.get("weight_tabular", 1.0))

    with open(combo_map_path, "r") as f:
        combo_map = json.load(f)

    ae = tf.keras.models.load_model(model_path, compile=False)

    # 2) FE: rebuild features (must match training FE)
    feats_train, tab_cols_train, _ = build_dataset_from_df(df_train_raw)
    feats_test, tab_cols_test, _ = build_dataset_from_df(df_test_raw)

    # Align tabular columns to those actually used in training
    tab_cols = [c for c in meta_tab_cols if c in feats_train.columns and c in feats_test.columns]

    # 3) Apply combo map
    feats_train = apply_loaded_combo_map(feats_train, combo_map)
    feats_test = apply_loaded_combo_map(feats_test, combo_map)

    # 4) Train-only combo stats and y_norm
    stats_train = build_combo_stats_train(feats_train)
    y_train = normalize_amount(feats_train, stats_train)
    y_test = normalize_amount(feats_test, stats_train)

    # 5) AE inputs
    Xtr, Ctr, col_names, j_y = build_inputs_with_ynorm(feats_train, tab_cols, y_train)
    Xte, Cte, _, _           = build_inputs_with_ynorm(feats_test, tab_cols, y_test)

    # 6) Column weights: length = n_tab + 1 (last is y_norm)
    n_tab = len(tab_cols)
    col_w = np.full(n_tab + 1, weight_tabular, dtype=np.float32)
    col_w[-1] = weight_y_norm

    state = dict(
        ae=ae,
        meta=meta,
        combo_map=combo_map,
        feats_train=feats_train,
        feats_test=feats_test,
        stats_train=stats_train,
        tab_cols=tab_cols,
        col_names=col_names,
        j_y=j_y,
        Xtr=Xtr,
        Ctr=Ctr,
        Xte=Xte,
        Cte=Cte,
        col_w=col_w,
    )
    return state


# ========= Core explanation ==========

def _map_focus_features(focus_features: Optional[List[str]],
                        col_names: List[str]) -> List[str]:
    """
    Map user-friendly names (e.g. 'amount') to actual column names (e.g. 'y_norm').
    Only keep ones that exist in col_names.
    """
    if not focus_features:
        return []

    alias = {
        "amount": "y_norm",
        "amount_norm": "y_norm",
        "amount_behavior": "y_norm",
    }
    mapped = []
    for f in focus_features:
        internal = alias.get(f, f)
        if internal in col_names:
            mapped.append(internal)
    mapped = list(dict.fromkeys(mapped))  # de-duplicate, keep order
    return mapped


def _compute_feature_contrib_for_row(state: Dict[str, Any],
                                     test_index: int) -> pd.DataFrame:
    """
    Compute per-feature reconstruction-error contribution for one test row.
    Returns a DataFrame with:
      feature, x_true, x_pred, diff, weight, contribution
    """
    Xte = state["Xte"]
    Cte = state["Cte"]
    col_names = state["col_names"]
    col_w = state["col_w"]
    ae = state["ae"]

    if test_index < 0 or test_index >= Xte.shape[0]:
        raise IndexError(f"test_index {test_index} out of range [0, {Xte.shape[0]})")

    x_true = Xte[test_index:test_index+1]  # shape (1, n_in)
    c_true = Cte[test_index:test_index+1]

    x_pred = ae.predict([x_true, c_true], batch_size=1, verbose=0)[0]  # shape (n_in,)

    diff = x_pred - x_true[0]
    se = diff ** 2
    contrib = se * col_w

    df = pd.DataFrame({
        "feature": col_names,
        "x_true": x_true[0],
        "x_pred": x_pred,
        "diff": diff,
        "squared_error": se,
        "weight": col_w,
        "contribution": contrib,
    })
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return df


def explain_single_anomaly_by_index(state: Dict[str, Any],
                                    test_index: int,
                                    focus_features: Optional[List[str]] = None,
                                    out_dir: str = "ae_anomaly_explanations",
                                    show: bool = True,
                                    top_k_if_no_focus: int = 6) -> str:
    """
    Explain one test row (by index) using per-feature reconstruction contributions.

    - focus_features: list of feature names to highlight (e.g. ["amount", "month", "cashbook_flag_derived"])
                      If empty/None, uses top_k_if_no_focus by contribution.
    - Returns: path to saved PNG.
    """
    feats_test = state["feats_test"]
    stats_train = state["stats_train"]
    col_names = state["col_names"]
    j_y = state["j_y"]

    df_contrib = _compute_feature_contrib_for_row(state, test_index)
    mapped_focus = _map_focus_features(focus_features, col_names)

    if mapped_focus:
        df_plot = df_contrib[df_contrib["feature"].isin(mapped_focus)].copy()
        if df_plot.empty:
            df_plot = df_contrib.head(top_k_if_no_focus).copy()
    else:
        df_plot = df_contrib.head(top_k_if_no_focus).copy()

    # Get raw amount + predicted amount for info
    row = feats_test.iloc[test_index]
    combo_str = row["combo_str"]
    acct, bu, code = combo_str.split("|")
    amount_actual = float(row["amount"])

    # compute y_norm_pred for this row and invert to amount_pred_inr
    Xte = state["Xte"]
    Cte = state["Cte"]
    ae = state["ae"]

    x_true = Xte[test_index:test_index+1]
    c_true = Cte[test_index:test_index+1]
    x_pred = ae.predict([x_true, c_true], batch_size=1, verbose=0)[0]
    y_norm_pred = x_pred[j_y:j_y+1]  # shape (1,)
    amount_pred = invert_pred_to_inr(feats_test.iloc[[test_index]].copy(), y_norm_pred, stats_train)[0]

    # Recon error for this row
    col_w = state["col_w"]
    recon_err = compute_row_recon_error(x_true, x_pred.reshape(1, -1), col_w)[0]

    # ================== Plot ==================
    _mkdir(out_dir)
    plt.figure(figsize=(8, 5))

    y_labels = df_plot["feature"].tolist()
    x_vals = df_plot["contribution"].values

    colors = []
    for f in df_plot["feature"]:
        if f in mapped_focus:
            colors.append("tab:red")
        else:
            colors.append("tab:blue")

    y_pos = np.arange(len(y_labels))
    plt.barh(y_pos, x_vals, color=colors, alpha=0.8)
    plt.yticks(y_pos, y_labels)
    plt.xlabel("Reconstruction-error contribution")
    plt.title(
        f"AE feature contributions (Acct={acct}, BU={bu}, Code={code})\n"
        f"Test idx={test_index}, ReconError={recon_err:.4f}, "
        f"Amount actual={amount_actual:.2f}, pred={amount_pred:.2f}"
    )
    plt.grid(axis="x", alpha=0.3)

    fname = f"ae_anomaly_contrib_idx{test_index}_{acct}_{bu}_{code}.png"
    fname = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in fname)
    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    # ========== Print small table (for notebook inspection) ==========
    print("\n[INFO] Feature-level contributions for test index", test_index)
    print(df_plot[["feature", "x_true", "x_pred", "diff", "contribution"]].to_string(index=False))

    print(f"\n[SAVED] {out_path}")
    return out_path


# ========= Explain all anomalies from CSV =========

def _build_anomaly_mask_from_csv(feats_test: pd.DataFrame,
                                 anomalies_csv_path: str) -> np.ndarray:
    """
    Build boolean mask over feats_test rows indicating which are anomalies,
    by matching on (ts date, combo_str, amount) key with anomalies CSV.
    """
    anoms = pd.read_csv(anomalies_csv_path)
    ft = feats_test.copy()

    ft["ts"] = pd.to_datetime(ft["ts"])
    anoms["ts"] = pd.to_datetime(anoms["ts"])

    if "combo_str" not in anoms.columns and {"BankAccountCode", "BusinessUnitCode", "BankTransactionCode"} <= set(anoms.columns):
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
    mask = ft["__key"].isin(anom_keys).values
    return mask


def explain_all_anomalies_from_csv(state: Dict[str, Any],
                                   anomalies_csv_path: str,
                                   focus_features: Optional[List[str]] = None,
                                   max_anomalies: int = 50,
                                   out_dir: str = "ae_anomaly_explanations",
                                   show: bool = False) -> List[str]:
    """
    Loop over all anomalies (as per anomalies CSV) and generate one plot per anomaly row.
    Returns list of saved PNG paths.
    """
    feats_test = state["feats_test"]

    anomaly_mask = _build_anomaly_mask_from_csv(feats_test, anomalies_csv_path)
    idx_all = np.where(anomaly_mask)[0]

    if len(idx_all) == 0:
        print("[INFO] No anomalies found in test set for explanation.")
        return []

    print(f"[INFO] Explaining up to {max_anomalies} anomalies (found {len(idx_all)}).")

    paths: List[str] = []
    for idx in idx_all[:max_anomalies]:
        p = explain_single_anomaly_by_index(
            state,
            test_index=int(idx),
            focus_features=focus_features,
            out_dir=out_dir,
            show=show,
        )
        paths.append(p)
    return paths


from ae_anomaly_feature_explainer import (
    prepare_explainer_state,
    explain_single_anomaly_by_index,
    explain_all_anomalies_from_csv,
)

# 1) Prepare state from your raw yearly DFs
state = prepare_explainer_state(df_train_year1, df_test_jan_feb)

# 2A) Explain one anomaly row (by index in test DF)
explain_single_anomaly_by_index(
    state,
    test_index=153,  # example
    focus_features=["amount", "month", "cashbook_flag_derived"],
    out_dir="ae_anomaly_explanations",
    show=True,
)

# 2B) Explain all anomalies from your AE CSV
paths = explain_all_anomalies_from_csv(
    state,
    anomalies_csv_path="combo_ae_outputs/anomalies_combo_ae.csv",
    focus_features=["amount", "month", "cashbook_flag_derived"],
    max_anomalies=30,
    out_dir="ae_anomaly_explanations",
    show=False,  # set True to spam plots in notebook
)

