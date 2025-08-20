import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SampleSizeConfig:
    bu_col: str = "BUnit"
    cp_col: str = "Cpty"
    # Analytical target: how tight should the mean reconstruction error estimate be?
    # e.g., 0.30 means ±30% of sigma at 95% confidence
    precision_rel: float = 0.30
    z_value: float = 1.96  # 95% CI
    min_floor: int = 30    # never recommend fewer than this (business-friendly)
    quarantine_floor: int = 7  # purely for reporting context
    # Empirical (optional) learning-curve
    do_empirical: bool = False
    max_empirical_k: int = 120
    step_k: int = 10
    bootstraps: int = 5
    probe_min: int = 5      # need at least this many holdout samples in the cell

class NewCellSampleSizer:
    def __init__(
        self,
        fit_fn: Callable,
        recon_error_fn: Callable,
        preprocess_fn: Optional[Callable] = None,
        config: SampleSizeConfig = SampleSizeConfig(),
        random_state: int = 42,
    ):
        self.fit_fn = fit_fn
        self.recon_error_fn = recon_error_fn
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else (lambda df: df)
        self.cfg = config
        self.rs = np.random.RandomState(random_state)

    def _train_and_score(self, train_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
        X_train = self.preprocess_fn(train_df)
        model = self.fit_fn(X_train)
        X_score = self.preprocess_fn(score_df)
        return self.recon_error_fn(model, X_score)

    def _baseline_stats(self, hist_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Train once on historical and compute reconstruction-error stats per BU and per CP."""
        errs_hist = self._train_and_score(hist_df, hist_df)
        df = hist_df.copy()
        df["_re"] = errs_hist
        global_sigma = df["_re"].std(ddof=1)
        bu_stats = df.groupby(self.cfg.bu_col)["_re"].agg(count="count", mean="mean", var="var")
        cp_stats = df.groupby(self.cfg.cp_col)["_re"].agg(count="count", mean="mean", var="var")
        return global_sigma, bu_stats, cp_stats

    def _analytical_n(self, bu: str, cp: str, bu_stats: pd.DataFrame, cp_stats: pd.DataFrame,
                      global_sigma: float) -> int:
        parts, weights = [], []
        if bu in bu_stats.index and np.isfinite(bu_stats.loc[bu, "var"]):
            parts.append(bu_stats.loc[bu, "var"])
            weights.append(max(int(bu_stats.loc[bu, "count"]), 1))
        if cp in cp_stats.index and np.isfinite(cp_stats.loc[cp, "var"]):
            parts.append(cp_stats.loc[cp, "var"])
            weights.append(max(int(cp_stats.loc[cp, "count"]), 1))

        if parts:
            sigma = np.sqrt(np.average(parts, weights=weights))
        else:
            sigma = float(global_sigma) if np.isfinite(global_sigma) else 1.0

        delta = max(self.cfg.precision_rel * sigma, 1e-12)
        n = int(np.ceil((self.cfg.z_value * sigma / delta) ** 2))
        return max(n, self.cfg.min_floor)

    def _empirical_curve(self, hist_df: pd.DataFrame, cell_df: pd.DataFrame) -> Optional[Dict]:
        """Optional: learning curve by injecting k samples from the cell and measuring probe error."""
        n_total = len(cell_df)
        # Need at least probe_min to evaluate anything
        if n_total < max(self.cfg.probe_min + self.cfg.step_k, self.cfg.min_floor):
            return None

        ks = []
        means = []
        stds = []

        # Precompute a fixed probe set: use the last probe_min (or rs choice)
        probe_idx = self.rs.choice(n_total, size=min(self.cfg.probe_min, n_total // 3), replace=False)
        probe_df = cell_df.iloc[probe_idx]
        remaining_idx = np.setdiff1d(np.arange(n_total), probe_idx)
        rem_df = cell_df.iloc[remaining_idx]

        max_k = min(self.cfg.max_empirical_k, len(remaining_idx))
        if max_k < self.cfg.step_k:
            return None

        for k in range(self.cfg.step_k, max_k + 1, self.cfg.step_k):
            # Bootstrap repeats
            boot_means = []
            for _ in range(self.cfg.bootstraps):
                take_idx = self.rs.choice(len(rem_df), size=k, replace=False)
                inj_df = rem_df.iloc[take_idx]
                train_df = pd.concat([hist_df, inj_df], axis=0)
                probe_errs = self._train_and_score(train_df, probe_df)
                boot_means.append(np.mean(probe_errs))

            ks.append(k)
            means.append(float(np.mean(boot_means)))
            stds.append(float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0)

        # Choose elbow: first k where relative improvement < 5% and CV < 0.2
        rec_n = None
        for i in range(1, len(ks)):
            prev, curr = means[i-1], means[i]
            rel_impr = (prev - curr) / max(prev, 1e-12)
            cv = stds[i] / max(curr, 1e-12)
            if (rel_impr < 0.05) and (cv < 0.20):
                rec_n = ks[i]
                break
        if rec_n is None:
            rec_n = ks[-1]

        return {
            "k_grid": ks,
            "mean_probe_err": means,
            "std_probe_err": stds,
            "empirical_recommended_n": int(rec_n),
        }

    def recommend(
        self,
        hist_df: pd.DataFrame,
        recent_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return per-cell recommended sample sizes for *new* BU×CP combos."""
        bu_col, cp_col = self.cfg.bu_col, self.cfg.cp_col
        # Identify new BU×CP combos absent in history
        hist_pairs = set(zip(hist_df[bu_col].astype(str), hist_df[cp_col].astype(str)))
        recent_pairs = set(zip(recent_df[bu_col].astype(str), recent_df[cp_col].astype(str)))
        new_pairs = sorted(list(recent_pairs - hist_pairs))

        global_sigma, bu_stats, cp_stats = self._baseline_stats(hist_df)

        rows = []
        for bu, cp in new_pairs:
            cell_df = recent_df[(recent_df[bu_col].astype(str) == bu) & (recent_df[cp_col].astype(str) == cp)]
            avail = len(cell_df)

            analytical_n = self._analytical_n(bu, cp, bu_stats, cp_stats, global_sigma)

            empirical_info = None
            empirical_n = None
            if self.cfg.do_empirical and avail >= self.cfg.min_floor + self.cfg.probe_min:
                empirical_info = self._empirical_curve(hist_df, cell_df)
                if empirical_info is not None:
                    empirical_n = empirical_info["empirical_recommended_n"]

            # Final recommendation: prefer the *max* of analytical and empirical (safer)
            recommended = int(max(analytical_n, empirical_n or 0))

            rows.append({
                "BU": bu,
                "CP": cp,
                "available_examples_now": avail,
                "analytical_required_n": analytical_n,
                "empirical_required_n": empirical_n,
                "recommended_required_n": recommended,
                "shortfall_vs_recommended": max(recommended - avail, 0),
            })

        out = pd.DataFrame(rows).sort_values(["recommended_required_n", "BU", "CP"]).reset_index(drop=True)

        # Add a one-row summary (typical target) as attributes for convenience
        if len(out) > 0:
            out.attrs["summary"] = {
                "median_recommended_n": int(np.median(out["recommended_required_n"])),
                "p80_recommended_n": int(np.percentile(out["recommended_required_n"], 80)),
                "rule_of_thumb": (
                    "Most new BU×CP cells will be adequately represented once they reach ~"
                    f"{int(np.median(out['recommended_required_n']))} samples "
                    "(95% CI ±{:.0f}%·σ target)".format(self.cfg.precision_rel * 100)
                ),
            }
        return out
#==================================


# 1) Provide your three functions
def preprocess_fn(df: pd.DataFrame):
    # return model-ready X (e.g., column transformer / one-hot with handle_unknown='ignore')
    return my_pipeline.transform(df)  # or .fit_transform on hist first in your code

def fit_fn(X, y=None, sample_weight=None):
    # train your autoencoder with FIXED hyper-params (same as production retrain)
    model = my_autoencoder_class(**fixed_hparams)
    model.fit(X)  # add sample_weight if you use it
    return model

def recon_error_fn(model, X) -> np.ndarray:
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)  # or your production error metric

# 2) Configure (analytical only, fast)
cfg = SampleSizeConfig(
    bu_col="BUnit", cp_col="Cpty",
    precision_rel=0.30,   # ±30%·σ at 95% CI → often ~40–60 samples
    do_empirical=False     # turn True to run learning-curve simulation too
)

sizer = NewCellSampleSizer(fit_fn, recon_error_fn, preprocess_fn, cfg)

# 3) Run
result_df = sizer.recommend(hist_df, recent_df)

# 4) Read the summary for business
print(result_df.attrs.get("summary", {}))
# -> {'median_recommended_n': 48, 'p80_recommended_n': 72, 'rule_of_thumb': 'Most new BU×CP cells ...'}
