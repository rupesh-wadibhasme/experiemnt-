# ==== DAILY COUNTS BASELINE (exact, month-partitioned) ======================
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

class DailyCountStore:
    """
    Rolling per-day txn counts by group.
    Stores month-partitioned parquet files: artifacts_features/baselines/daily_counts/YYYY-MM.parquet
    Columns: [BankAccountCode, BusinessUnitCode, BankTransactionCode, _day (date), count (int)]
    """
    def __init__(self, root="artifacts_features/baselines/daily_counts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_month(self, d: date) -> Path:
        return self.root / f"{d.strftime('%Y-%m')}.parquet"

    def _ensure_parquet_supported(self):
        try:
            pd.DataFrame({"x":[1]}).to_parquet(self.root / "_probe.parquet", index=False)
            (self.root / "_probe.parquet").unlink(missing_ok=True)
        except Exception as e:
            raise RuntimeError("Parquet support required (install pyarrow or fastparquet).") from e

    def upsert_counts(self, df_today: pd.DataFrame,
                      keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                      posting_col="PostingDateKey") -> None:
        """Append/replace today’s counts for each group. Call AFTER spike check to avoid leakage."""
        self._ensure_parquet_supported()
        day = pd.to_datetime(df_today[posting_col].astype(str), format="%Y%m%d", errors="coerce").dt.date
        tmp = df_today.assign(_day=day)
        counts = (tmp.groupby([*keys, "_day"], dropna=False)
                    .size().reset_index(name="count"))
        if counts.empty:
            return
        path = self._path_for_month(counts["_day"].iloc[0])
        if path.exists():
            old = pd.read_parquet(path)
            merged = (pd.concat([old, counts], ignore_index=True)
                        .drop_duplicates(subset=[*keys, "_day"], keep="last"))
        else:
            merged = counts
        merged.to_parquet(path, index=False)

    def load_history(self, groups_df: pd.DataFrame, days_back=30,
                     keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                     posting_col="PostingDateKey") -> pd.DataFrame:
        """Load [today - days_back, yesterday] per-day counts for groups present in groups_df."""
        self._ensure_parquet_supported()
        today_vals = pd.to_datetime(groups_df[posting_col].astype(str), format="%Y%m%d", errors="coerce").dt.date
        end = max([d for d in today_vals if pd.notna(pd.Timestamp(d))], default=None)
        if end is None:
            return pd.DataFrame(columns=[*keys, "_day", "count"])
        start = end - timedelta(days=days_back)
        # Which months to pull
        months = pd.period_range(start=start, end=end, freq="M").strftime("%Y-%m").tolist()
        months = sorted(set(months + [start.strftime("%Y-%m"), end.strftime("%Y-%m")]))
        parts = []
        for m in months:
            p = self.root / f"{m}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        if not parts:
            return pd.DataFrame(columns=[*keys, "_day", "count"])
        hist = pd.concat(parts, ignore_index=True)
        hist = hist[(hist["_day"] >= start) & (hist["_day"] <= end)]
        # keep only groups we care about today
        grp_today = groups_df[[*keys]].drop_duplicates()
        hist = hist.merge(grp_today, on=list(keys), how="inner")
        return hist

# ==== SPIKE RULE (z-score or percentile; leakage-free via trailing shift) ===
def flag_spikes_from_store(df_today: pd.DataFrame,
                           store: DailyCountStore,
                           keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                           posting_col="PostingDateKey",
                           method="zscore",
                           horizon_days=30,
                           z_k=3.0,
                           pct=0.99,
                           min_active_days=7) -> pd.DataFrame:
    """
    Returns df_today with added columns:
      - group_count_today, hist_mean_30d, hist_std_30d, hist_pctl_30d, hist_active_days
      - vol_spike_flag (0/1), vol_spike_reason (str)
    """
    # Today’s per-group count
    pdt = pd.to_datetime(df_today[posting_col].astype(str), format="%Y%m%d", errors="coerce").dt.date
    today_counts = (df_today.assign(_day=pdt)
                            .groupby([*keys, "_day"], dropna=False)
                            .size().reset_index(name="group_count_today"))
    if today_counts.empty:
        out = df_today.copy()
        out["group_count_today"] = 0
        out["hist_mean_30d"] = 0.0
        out["hist_std_30d"] = 0.0
        out["hist_pctl_30d"] = 0.0
        out["hist_active_days"] = 0
        out["vol_spike_flag"] = 0
        out["vol_spike_reason"] = ""
        return out

    # Load history (<= yesterday)
    hist = store.load_history(df_today, days_back=horizon_days, keys=keys, posting_col=posting_col)

    if hist.empty:
        joined = today_counts.assign(hist_mean_30d=0.0, hist_std_30d=0.0,
                                     hist_pctl_30d=0.0, hist_active_days=0)
    else:
        # Build continuous timeline per group, then rolling stats shifted(1)
        def _roll(g):
            idx = pd.date_range(g["_day"].min(), g["_day"].max(), freq="D").date
            tmp = pd.DataFrame({"_day": idx}).merge(g[["_day","count"]], on="_day", how="left")
            tmp["count"] = tmp["count"].fillna(0)
            roll = tmp["count"].rolling(window=horizon_days, min_periods=1)
            mean_ = roll.mean().shift(1)
            std_  = roll.std(ddof=1).shift(1)
            pctl  = roll.apply(lambda x: np.quantile(x, pct), raw=False).shift(1)
            active = roll.apply(lambda x: int(np.sum(np.array(x) > 0)), raw=False).shift(1)
            out = tmp.copy()
            out["hist_mean_30d"] = mean_.values
            out["hist_std_30d"]  = std_.values
            out["hist_pctl_30d"] = pctl.values
            out["hist_active_days"] = active.values
            # keep only original days (when group had activity historically)
            return out[out["_day"].isin(g["_day"])]

        rolled = (hist.groupby(list(keys), group_keys=False)
                       .apply(_roll)
                       .reset_index())

        joined = today_counts.merge(
            rolled.rename(columns={"count":"count_hist"}),
            on=[*keys, "_day"], how="left"
        )
        for c in ["hist_mean_30d","hist_std_30d","hist_pctl_30d","hist_active_days"]:
            joined[c] = joined[c].fillna(0)

    # Decision
    has_hist = joined["hist_active_days"] >= min_active_days
    if method == "zscore":
        std0 = (joined["hist_std_30d"] == 0) & (joined["group_count_today"] > joined["hist_mean_30d"])
        stdk = (joined["hist_std_30d"] > 0) & (
            joined["group_count_today"] > joined["hist_mean_30d"] + z_k * joined["hist_std_30d"]
        )
        flag = has_hist & (std0 | stdk)
        thresh = np.where(
            std0,
            (joined["hist_mean_30d"]).round(2).astype(str) + " (std≈0)",
            (joined["hist_mean_30d"] + z_k * joined["hist_std_30d"]).round(2).astype(str)
        )
    else:
        pthr = joined["hist_pctl_30d"]
        flag = has_hist & (joined["group_count_today"] > pthr)
        thresh = pthr.round(2).astype(str)

    joined["vol_spike_flag"] = flag.astype(int)
    joined["vol_spike_reason"] = [
        f"Txn volume spike: {int(c)} today vs typical {t} "
        f"for Account='{a}', BU='{b}', Code='{cd}' (last {horizon_days}d)."
        for c, t, a, b, cd in zip(
            joined["group_count_today"], thresh,
            joined[keys[0]], joined[keys[1]], joined[keys[2]]
        )
    ]

    # Attach back to each row
    out = df_today.copy()
    out["_day"] = pdt
    out = out.merge(
        joined[[*keys, "_day", "group_count_today", "hist_mean_30d","hist_std_30d",
                "hist_pctl_30d","hist_active_days","vol_spike_flag","vol_spike_reason"]],
        on=[*keys, "_day"], how="left"
    ).drop(columns=["_day"])
    # defaults
    for c in ["group_count_today","hist_mean_30d","hist_std_30d","hist_pctl_30d","hist_active_days","vol_spike_flag"]:
        out[c] = out[c].fillna(0)
    out["vol_spike_reason"] = out["vol_spike_reason"].fillna("")
    return out

# ==== NON-BUSINESS DAY RULE (weekend/holiday) ===============================
def flag_non_business_day_rules(df: pd.DataFrame,
                                value_col="ValueDateKey",
                                posting_col="PostingDateKey",
                                holidays=None) -> pd.DataFrame:
    """
    Adds: is_nonbiz_value, is_nonbiz_post, nonbiz_reason
    holidays: set({date(YYYY,MM,DD), ...}) or None
    """
    holidays = holidays or set()
    out = df.copy()

    def _parse(s):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")

    v = _parse(out[value_col]) if value_col in out.columns else pd.Series(pd.NaT, index=out.index)
    p = _parse(out[posting_col]) if posting_col in out.columns else pd.Series(pd.NaT, index=out.index)

    is_wknd_v = v.dt.dayofweek.isin([5,6])
    is_wknd_p = p.dt.dayofweek.isin([5,6])

    is_hol_v = v.dt.date.isin(holidays) if len(holidays)>0 else pd.Series(False, index=out.index)
    is_hol_p = p.dt.date.isin(holidays) if len(holidays)>0 else pd.Series(False, index=out.index)

    out["is_nonbiz_value"] = (is_wknd_v | is_hol_v).astype(int)
    out["is_nonbiz_post"]  = (is_wknd_p | is_hol_p).astype(int)

    reasons = []
    for i in range(len(out)):
        r = ""
        if pd.notna(p.iloc[i]):
            if is_wknd_p.iloc[i] or (len(holidays)>0 and is_hol_p.iloc[i]):
                typ = "weekend" if is_wknd_p.iloc[i] else "holiday"
                r = f"PostingDate {p.iloc[i].date()} is a non-business day ({typ})."
        if r == "" and pd.notna(v.iloc[i]):
            if is_wknd_v.iloc[i] or (len(holidays)>0 and is_hol_v.iloc[i]):
                typ = "weekend" if is_wknd_v.iloc[i] else "holiday"
                r = f"ValueDate {v.iloc[i].date()} is a non-business day ({typ})."
        reasons.append(r)
    out["nonbiz_reason"] = reasons
    return out
