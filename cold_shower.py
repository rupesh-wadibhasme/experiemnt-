# sample_size_helper_plain.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, List, Tuple

# --------------------------
# Config (edit as you like)
# --------------------------
DEFAULT_CONFIG = {
    # Core ID columns
    "bu_col": "BUnit",
    "cp_col": "Cpty",

    # ---- Absolute precision target (error units, e.g., MSE units) ----
    # "Estimate the mean reconstruction error within ±precision_abs at 95% confidence"
    "precision_abs": 0.25,
    "z_value": 1.96,          # 95% CI

    # Floors / limits
    "min_floor": 30,          # never recommend below this
    "quarantine_floor": 7,    # informational only (not used in n calc)

    # ---- Empirical learning-curve (validator; requires retraining) ----
    "do_empirical": True,     # set False to skip simulation validator
    "step_k": 10,             # inject 10, 20, 30, ...
    "max_empirical_k": 200,   # cap on k
    "bootstraps": 5,          # repeats per k
    "probe_min": 8,           # held-out probe from the new cell
    "elbow_rel_impr": 0.05,   # stop when relative improvement < 5%
    "elbow_cv": 0.20,         # and CV < 0.20

    # Reproducibility
    "random_state": 42,

    # -------- Feature-aware drift controls (beyond BU×CP) --------
    # Numeric columns whose scale/shape changes should affect required n
    "numeric_cols": ["BuyAmount", "SellAmount"],

    # Categorical columns used to select a matched cohort in history
    "match_cols": ["Instrument", "BuyCurr", "SellCurr"],

    # PSI settings & drift-to-inflation mapping
    "psi_bins": 10,
    "min_recent_for_drift": 5,
    "vif_cap": 3.0,  # cap total sigma inflation at ×3 (tune as needed)
}

# --------------------------------
# Core utility functions (no OOP)
# --------------------------------

def train_and_score(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    train_fn: Callable[[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], object],
    preprocess_fn: Callable[[pd.DataFrame], np.ndarray],
    recon_error_fn: Callable[[object, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Fit model (fixed hyper-params) on train_df and return per-row errors for score_df."""
    X_train = preprocess_fn(train_df)
    model = train_fn(X_train, None, None)   # y and sample_weight optional
    X_score = preprocess_fn(score_df)
    errs = recon_error_fn(model, X_score)
    errs = np.asarray(errs).reshape(-1)
    return errs

def baseline_stats(
    hist_df: pd.DataFrame,
    bu_col: str,
    cp_col: str,
    train_fn: Callable,
    preprocess_fn: Callable,
    recon_error_fn: Callable,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Train once on history and compute global, BU, CP error stats."""
    errs_hist = train_and_score(hist_df, hist_df, train_fn, preprocess_fn, recon_error_fn)
    dfh = hist_df.copy()
    dfh["_re"] = errs_hist
    global_sigma = float(dfh["_re"].std(ddof=1))
    bu_stats = dfh.groupby(bu_col)["_re"].agg(count="count", mean="mean", var="var")
    cp_stats = dfh.groupby(cp_col)["_re"].agg(count="count", mean="mean", var="var")
    return global_sigma, bu_stats, cp_stats

def pooled_sigma(
    bu: str,
    cp: str,
    bu_stats: pd.DataFrame,
    cp_stats: pd.DataFrame,
    global_sigma: float,
) -> float:
    """Weighted-average variance from BU and CP marginals; fallback to global sigma."""
    parts, weights = [], []
    if bu in bu_stats.index and np.isfinite(bu_stats.loc[bu, "var"]):
        parts.append(float(bu_stats.loc[bu, "var"]))
        weights.append(max(int(bu_stats.loc[bu, "count"]), 1))
    if cp in cp_stats.index and np.isfinite(cp_stats.loc[cp, "var"]):
        parts.append(float(cp_stats.loc[cp, "var"]))
        weights.append(max(int(cp_stats.loc[cp, "count"]), 1))

    if parts:
        var = float(np.average(parts, weights=weights))
        sigma = float(np.sqrt(max(var, 0.0)))
    else:
        sigma = float(global_sigma) if np.isfinite(global_sigma) else 1.0
    return sigma

def analytical_n_abs(
    sigma_cell: float,
    precision_abs: float,
    z_value: float,
    min_floor: int,
) -> int:
    """Absolute-precision sample size: n = ceil((z*sigma / delta)^2)."""
    delta = max(float(precision_abs), 1e-12)
    n = int(np.ceil((float(z_value) * float(sigma_cell) / delta) ** 2))
    return max(n, int(min_floor))

# ----------------------------------------------------------
# Feature-aware drift: cohort matching, PSI, variance ratio
# ----------------------------------------------------------

def get_matched_history(hist_df: pd.DataFrame, cell_df: pd.DataFrame, match_cols: List[str]) -> pd.DataFrame:
    """Return historical rows matching the modal values of match_cols in the new cell."""
    if not match_cols:
        return hist_df
    subset = hist_df
    for col in match_cols:
        if col in hist_df.columns and col in cell_df.columns:
            val = cell_df[col].mode(dropna=True)
            if len(val) == 0:
                continue
            subset = subset[subset[col].astype(str) == str(val.iloc[0])]
    return subset if len(subset) else hist_df  # fallback to all history

def psi_1d(x_ref: np.ndarray, x_new: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between reference and new numeric arrays (equal-frequency bins on ref)."""
    x_ref = np.asarray(x_ref); x_new = np.asarray(x_new)
    x_ref = x_ref[np.isfinite(x_ref)]
    x_new = x_new[np.isfinite(x_new)]
    if len(x_ref) < 10 or len(x_new) < 2:
        return 0.0
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(x_ref, qs))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(x_ref, bins=edges)
    new_hist, _ = np.histogram(x_new, bins=edges)
    ref_p = ref_hist / max(ref_hist.sum(), 1)
    new_p = new_hist / max(new_hist.sum(), 1)
    eps = 1e-6
    psi = np.sum((new_p - ref_p) * np.log((new_p + eps) / (ref_p + eps)))
    return float(abs(psi))

def variance_ratio_1d(x_ref: np.ndarray, x_new: np.ndarray) -> float:
    """Return sqrt(var_new/var_ref), floored at 1 (scale inflation only)."""
    x_ref = np.asarray(x_ref); x_new = np.asarray(x_new)
    x_ref = x_ref[np.isfinite(x_ref)]
    x_new = x_new[np.isfinite(x_new)]
    if len(x_ref) < 5 or len(x_new) < 2:
        return 1.0
    v_ref = np.var(x_ref, ddof=1)
    v_new = np.var(x_new, ddof=1)
    if v_ref <= 0:
        return 1.0
    ratio = np.sqrt(max(v_new / v_ref, 1.0))
    return float(ratio)

def psi_to_factor(psi: float) -> float:
    """Map PSI to an inflation factor (coarse finance-standard buckets)."""
    if psi < 0.10: return 1.0
    if psi < 0.25: return 1.2
    if psi < 0.50: return 1.5
    return 2.0

def numeric_vif(
    hist_match: pd.DataFrame,
    cell_df: pd.DataFrame,
    numeric_cols: List[str],
    psi_bins: int,
    min_recent: int,
    vif_cap: float,
) -> float:
    """
    Combine variance-ratio and PSI across numeric_cols into a single inflation factor.
    Uses geometric mean of per-feature factors; caps at vif_cap.
    """
    if not numeric_cols or len(cell_df) < max(min_recent, 2):
        return 1.0

    factors = []
    for col in numeric_cols:
        if col not in hist_match.columns or col not in cell_df.columns:
            continue
        ref = hist_match[col].values
        new = cell_df[col].values

        vr = variance_ratio_1d(ref, new)          # scale change
        p = psi_1d(ref, new, bins=psi_bins)       # shape change
        pf = psi_to_factor(p)

        f = max(vr, pf)  # conservative per-feature
        factors.append(f)

    if not factors:
        return 1.0
    gmean = float(np.exp(np.mean(np.log(np.maximum(factors, 1.0)))))
    return float(min(gmean, vif_cap))

# -----------------------------
# Empirical validator (curve)
# -----------------------------

def empirical_curve(
    hist_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    train_fn: Callable,
    preprocess_fn: Callable,
    recon_error_fn: Callable,
    cfg: Dict,
) -> Optional[Dict[str, List[float]]]:
    """
    Learning curve by injecting k samples (k=step_k,2*step_k,...) from cell_df into history,
    retraining each time, and scoring a small probe set held out from the cell.
    Returns a dict with k_grid, mean_probe_err, std_probe_err, and empirical_recommended_n.
    """
    rs = np.random.RandomState(cfg.get("random_state", 42))
    step_k = int(cfg.get("step_k", 10))
    max_empirical_k = int(cfg.get("max_empirical_k", 200))
    bootstraps = int(cfg.get("bootstraps", 5))
    probe_min = int(cfg.get("probe_min", 8))
    elbow_rel_impr = float(cfg.get("elbow_rel_impr", 0.05))
    elbow_cv = float(cfg.get("elbow_cv", 0.20))
    min_floor = int(cfg.get("min_floor", 30))

    n_total = len(cell_df)
    if n_total < max(probe_min + step_k, min_floor):
        return None

    # Probe split
    probe_size = min(probe_min, max(n_total // 3, probe_min))
    probe_idx = rs.choice(n_total, size=probe_size, replace=False)
    probe_df = cell_df.iloc[probe_idx]
    rem_idx = np.setdiff1d(np.arange(n_total), probe_idx)
    pool_df = cell_df.iloc[rem_idx]

    # k grid
    max_k = min(max_empirical_k, len(pool_df))
    if max_k < step_k:
        return None

    ks, means, stds = [], [], []
    for k in range(step_k, max_k + 1, step_k):
        boot_means = []
        for _ in range(bootstraps):
            take_idx = rs.choice(len(pool_df), size=k, replace=False)
            inj_df = pool_df.iloc[take_idx]
            train_df = pd.concat([hist_df, inj_df], axis=0)
            probe_errs = train_and_score(train_df, probe_df, train_fn, preprocess_fn, recon_error_fn)
            boot_means.append(float(np.mean(probe_errs)))
        mu = float(np.mean(boot_means))
        sd = float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0
        ks.append(k); means.append(mu); stds.append(sd)

    # Elbow selection: first point where improvement and variability are small
    rec_n = None
    for i in range(1, len(ks)):
        prev, curr = means[i-1], means[i]
        rel_impr = (prev - curr) / max(prev, 1e-12)
        cv = stds[i] / max(curr, 1e-12)
        if (rel_impr < elbow_rel_impr) and (cv < elbow_cv):
            rec_n = ks[i]
            break
    if rec_n is None:
        rec_n = ks[-1]  # conservative fallback

    return {
        "k_grid": ks,
        "mean_probe_err": means,
        "std_probe_err": stds,
        "empirical_recommended_n": int(rec_n),
    }

# -----------------------
# Recommendation driver
# -----------------------

def recommend(
    hist_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    train_fn: Callable,
    preprocess_fn: Callable,
    recon_error_fn: Callable,
    cfg: Dict = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    For each *new* BU×CP in recent_df (not present in hist_df), return:
      available_examples_now, sigma_cell, analytical_required_n_abs,
      empirical_required_n (if enabled/possible), recommended_required_n, shortfall_vs_recommended.
    Also returns a small summary dict.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    bu_col, cp_col = cfg.get("bu_col", "BUnit"), cfg.get("cp_col", "Cpty")

    # Identify new pairs
    hist_pairs = set(zip(hist_df[bu_col].astype(str), hist_df[cp_col].astype(str)))
    recent_pairs = set(zip(recent_df[bu_col].astype(str), recent_df[cp_col].astype(str)))
    new_pairs = sorted(list(recent_pairs - hist_pairs))

    # Baseline stats from history
    global_sigma, bu_stats, cp_stats = baseline_stats(
        hist_df, bu_col, cp_col, train_fn, preprocess_fn, recon_error_fn
    )

    rows = []
    curve_cache: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for bu, cp in new_pairs:
        cell_df = recent_df[(recent_df[bu_col].astype(str) == bu) &
                            (recent_df[cp_col].astype(str) == cp)]
        avail = int(len(cell_df))

        # Base sigma from BU/CP marginals
        sigma_cell = pooled_sigma(bu, cp, bu_stats, cp_stats, global_sigma)

        # ---- Feature-aware inflation using matched cohort + numeric drift ----
        hist_match = get_matched_history(hist_df, cell_df, cfg.get("match_cols", []))
        vif = numeric_vif(
            hist_match=hist_match,
            cell_df=cell_df,
            numeric_cols=cfg.get("numeric_cols", []),
            psi_bins=int(cfg.get("psi_bins", 10)),
            min_recent=int(cfg.get("min_recent_for_drift", 5)),
            vif_cap=float(cfg.get("vif_cap", 3.0)),
        )
        sigma_eff = sigma_cell * vif

        # Analytical target (absolute precision)
        analytical_n = analytical_n_abs(
            sigma_cell=sigma_eff,
            precision_abs=cfg.get("precision_abs", 0.25),
            z_value=cfg.get("z_value", 1.96),
            min_floor=cfg.get("min_floor", 30),
        )

        # Optional empirical validator
        empirical_n = None
        if cfg.get("do_empirical", True):
            curve = empirical_curve(
                hist_df=hist_df,
                cell_df=cell_df,
                train_fn=train_fn,
                preprocess_fn=preprocess_fn,
                recon_error_fn=recon_error_fn,
                cfg=cfg,
            )
            if curve is not None:
                empirical_n = int(curve["empirical_recommended_n"])
                curve_cache[(bu, cp)] = curve

        recommended = int(max(analytical_n, empirical_n or 0))
        rows.append({
            "BU": bu,
            "CP": cp,
            "available_examples_now": avail,
            "sigma_cell_raw": round(float(sigma_cell), 6),
            "sigma_inflation_vif": round(float(vif), 4),
            "sigma_cell_effective": round(float(sigma_eff), 6),
            "analytical_required_n_abs": int(analytical_n),
            "empirical_required_n": int(empirical_n) if empirical_n is not None else None,
            "recommended_required_n": recommended,
            "shortfall_vs_recommended": max(recommended - avail, 0),
        })

    df = pd.DataFrame(rows).sort_values(
        ["recommended_required_n", "BU", "CP"]
    ).reset_index(drop=True)

    summary = {}
    if not df.empty:
        rrn = df["recommended_required_n"].dropna().values
        summary = {
            "median_recommended_n": int(np.median(rrn)),
            "p80_recommended_n": int(np.percentile(rrn, 80)),
            "precision_abs": float(cfg.get("precision_abs", 0.25)),
            "z_value": float(cfg.get("z_value", 1.96)),
            "note": (
                "Analytical target uses absolute precision and feature-aware sigma inflation; "
                "empirical validator used when enough cell samples exist."
            ),
        }
    return df, summary

# -------------------
# Plotting helpers
# -------------------

def plot_learning_curve(curve: Dict[str, List[float]], title: str = "Empirical learning curve"):
    """Mean probe error vs. injected samples; shows std band and chosen elbow."""
    ks = curve["k_grid"]
    means = curve["mean_probe_err"]
    stds = curve["std_probe_err"]
    elbow = curve["empirical_recommended_n"]

    plt.figure()
    plt.plot(ks, means, marker="o")
    upper = np.array(means) + np.array(stds)
    lower = np.array(means) - np.array(stds)
    plt.fill_between(ks, lower, upper, alpha=0.2)
    if elbow is not None:
        i = ks.index(elbow)
        plt.scatter([elbow], [means[i]])
        plt.axvline(elbow, linestyle="--")
    plt.title(title)
    plt.xlabel("Injected samples (k)")
    plt.ylabel("Mean probe reconstruction error")
    plt.tight_layout()
    plt.show()

def plot_targets_bar(df: pd.DataFrame, top_n: int = 15, title: str = "Top new pairs by required samples"):
    """Bar plot of recommended_required_n for top N new BU×CP pairs."""
    if df.empty:
        return
    dfx = df.sort_values("recommended_required_n", ascending=False).head(top_n)
    labels = [f"{r.BU}×{r.CP}" for r in dfx.itertuples(index=False)]
    vals = dfx["recommended_required_n"].values

    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("New BU×CP")
    plt.ylabel("Recommended sample size")
    plt.tight_layout()
    plt.show()

def export_csv(df: pd.DataFrame, path: str = "sample_size_recommendations.csv"):
    """Export recommendations to CSV (underscores in filename)."""
    df.to_csv(path, index=False)

# -------------------
# Minimal usage demo
# -------------------
if __name__ == "__main__":
    # You MUST provide these three functions in your project:
    def preprocess_fn(df: pd.DataFrame) -> np.ndarray:
        """
        Return a dense float32 array with the same feature order used by the autoencoder.
        IMPORTANT: Fit any transformers on history only; here you should call pipeline.transform(df).
        """
        raise NotImplementedError("Provide preprocess_fn(df) -> np.ndarray")

    def train_fn(X: np.ndarray, y=None, sample_weight=None):
        """
        Train your autoencoder with fixed hyper-params and return the model.
        This is invoked repeatedly during empirical simulation.
        """
        raise NotImplementedError("Provide train_fn(X) -> model")

    def recon_error_fn(model, X: np.ndarray) -> np.ndarray:
        """
        Return per-row reconstruction error (shape: (n_samples,)).
        """
        raise NotImplementedError("Provide recon_error_fn(model, X) -> np.ndarray")

    # Example (replace with real dfs and provided fns):
    # hist_df = pd.read_parquet("history.parquet")
    # recent_df = pd.read_parquet("recent_2months.parquet")
    # cfg = {**DEFAULT_CONFIG,
    #        "precision_abs": 0.25,
    #        "do_empirical": True,
    #        "numeric_cols": ["BuyAmount","SellAmount"],
    #        "match_cols": ["Instrument","BuyCurr","SellCurr"]}
    # recs, summary = recommend(hist_df, recent_df, train_fn, preprocess_fn, recon_error_fn, cfg)
    # print(recs.head())
    # print(summary)
    # plot_targets_bar(recs, top_n=15)
    # # For a specific pair:
    # pair_mask = (recent_df[cfg["bu_col"]].astype(str)=="HSBC") & (recent_df[cfg["cp_col"]].astype(str)=="ANZ")
    # cell_df = recent_df[pair_mask]
    # curve = empirical_curve(hist_df, cell_df, train_fn, preprocess_fn, recon_error_fn, cfg)
    # if curve is not None:
    #     plot_learning_curve(curve, title="HSBC×ANZ: empirical learning curve")
    #     print({"empirical_recommended_n": curve["empirical_recommended_n"]})
