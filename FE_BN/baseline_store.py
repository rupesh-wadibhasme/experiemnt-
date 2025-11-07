# ==== DAILY COUNTS BASELINE (exact, month-partitioned) ======================
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

class DailyCountStore:
    """
    Lightweight in-memory/CSV store of daily counts per (acct, bu, code).
    Designed to work on the ENGINEERED DF (feats_*) that still has RAW columns,
    NOT on the encoded X matrix.
    """
    def __init__(self, path_csv: str = "artifacts_features/daily_counts_store.csv"):
        self.path_csv = path_csv
        self._df = None
        if os.path.exists(self.path_csv):
            self._df = pd.read_csv(self.path_csv, parse_dates=["_day"])
        else:
            self._df = pd.DataFrame(columns=[
                "BankAccountCode","BusinessUnitCode","BankTransactionCode","_day","count"
            ])
            self._df["_day"] = pd.to_datetime(self._df["_day"])

    @staticmethod
    def _ensure_raw_keys(df: pd.DataFrame, keys):
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise KeyError(
                f"DailyCountStore expected raw keys {keys} in the provided dataframe, "
                f"but missing {missing}. Make sure you pass the ENGINEERED DF (feats_*), "
                f"not the encoded matrix. Available columns: {list(df.columns)[:40]}..."
            )

    @staticmethod
    def _to_date(series_yyyymmdd: pd.Series) -> pd.Series:
        return pd.to_datetime(series_yyyymmdd.astype(str), format="%Y%m%d", errors="coerce").dt.date

    def upsert_counts(self,
                      df_today: pd.DataFrame,
                      keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                      posting_col="PostingDateKey") -> None:
        """
        Add/replace counts for the (keys, posting_date) combinations present in df_today.
        df_today MUST have the raw key columns (not hashed/one-hot) and raw posting_col.
        """
        self._ensure_raw_keys(df_today, keys)
        if posting_col not in df_today.columns:
            raise KeyError(f"'{posting_col}' not found in dataframe.")

        grp = (df_today.assign(_day=self._to_date(df_today[posting_col]))
                        .groupby([*keys, "_day"], dropna=False)
                        .size()
                        .reset_index(name="count"))

        if grp.empty:
            return

        # Remove any existing rows for these (keys,_day) then append new
        m = ["BankAccountCode","BusinessUnitCode","BankTransactionCode","_day"]
        cur = self._df
        left = cur.merge(grp[m], on=m, how="left", indicator=True)
        cur_pruned = left[left["_merge"] == "left_only"].drop(columns=["_merge"])
        cur_pruned = cur_pruned[cur_pruned.columns]  # keep same column order

        self._df = pd.concat([cur_pruned, grp], ignore_index=True)
        # Persist
        os.makedirs(os.path.dirname(self.path_csv), exist_ok=True)
        self._df.to_csv(self.path_csv, index=False)

    def load_history(self,
                     df_today: pd.DataFrame,
                     days_back: int = 30,
                     keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                     posting_col="PostingDateKey") -> pd.DataFrame:
        """
        Return historical rows for the (keys) appearing in df_today within the trailing window.
        """
        self._ensure_raw_keys(df_today, keys)
        tdates = self._to_date(df_today[posting_col])
        if len(tdates) == 0 or tdates.isna().all():
            return pd.DataFrame(columns=[*keys, "_day", "count"])

        day_max = pd.Timestamp(tdates.max())
        start = (day_max - pd.Timedelta(days=days_back)).normalize()
        end   = (day_max - pd.Timedelta(days=1)).normalize()

        if self._df is None or self._df.empty:
            return pd.DataFrame(columns=[*keys, "_day", "count"])

        key_df = df_today[[*keys]].drop_duplicates()
        hist = self._df.copy()
        hist["_day"] = pd.to_datetime(hist["_day"]).dt.normalize()

        # Inner join on keys to keep only groups that appear today
        hist = key_df.merge(hist, on=list(keys), how="inner")

        mask = (hist["_day"] >= start) & (hist["_day"] <= end)
        hist = hist.loc[mask, [*keys, "_day", "count"]]
        return hist

# ==== SPIKE RULE (z-score or percentile; leakage-free via trailing shift) ===
def flag_spikes_from_store(df_today: pd.DataFrame,
                           store: "DailyCountStore",
                           keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
                           posting_col="PostingDateKey",
                           method="zscore",
                           horizon_days=30,
                           z_k=3.0,
                           pct=0.99,
                           min_active_days=7) -> pd.DataFrame:
    """
    Non-recursive implementation.
    Adds to df_today:
      - group_count_today, hist_mean_30d, hist_std_30d, hist_pctl_30d, hist_active_days
      - vol_spike_flag (0/1), vol_spike_reason (str)
    """
    # --- today’s per-group count ---
    pdt = pd.to_datetime(df_today[posting_col].astype(str), format="%Y%m%d", errors="coerce").dt.date
    today_counts = (df_today.assign(_day=pdt)
                            .groupby([*keys, "_day"], dropna=False)
                            .size().reset_index(name="group_count_today"))

    # prepare output skeleton joined on group + day
    out = df_today.copy()
    out["_day"] = pdt

    if today_counts.empty:
        out["group_count_today"] = 0
        out["hist_mean_30d"] = 0.0
        out["hist_std_30d"] = 0.0
        out["hist_pctl_30d"] = 0.0
        out["hist_active_days"] = 0
        out["vol_spike_flag"] = 0
        out["vol_spike_reason"] = ""
        return out.drop(columns=["_day"])

    # --- load trailing history (<= yesterday) for just today’s groups ---
    hist = store.load_history(df_today, days_back=horizon_days, keys=keys, posting_col=posting_col)
    # ensure correct dtypes
    if not hist.empty:
        hist = hist.copy()
        # keep only required cols
        keep = [*keys, "_day", "count"]
        hist = hist[keep]

    # container for per-group baselines for today
    base_rows = []

    # iterate over today's unique groups only (usually small)
    for (a, b, c), g_today in today_counts.groupby(keys):
        day_today = g_today["_day"].iloc[0]
        cnt_today = int(g_today["group_count_today"].iloc[0])

        # slice history for this group
        if hist.empty:
            g_hist = pd.DataFrame(columns=["_day","count"])
        else:
            mask = (hist[keys[0]] == a) & (hist[keys[1]] == b) & (hist[keys[2]] == c)
            g_hist = hist.loc[mask, ["_day","count"]].copy()

        # build continuous calendar window [day_today - horizon_days, day_today - 1]
        start = pd.Timestamp(day_today) - pd.Timedelta(days=horizon_days)
        end   = pd.Timestamp(day_today) - pd.Timedelta(days=1)
        if end < start:
            # no prior day (edge case)
            base_rows.append({
                keys[0]: a, keys[1]: b, keys[2]: c, "_day": day_today,
                "group_count_today": cnt_today,
                "hist_mean_30d": 0.0, "hist_std_30d": 0.0, "hist_pctl_30d": 0.0, "hist_active_days": 0
            })
            continue

        idx = pd.date_range(start, end, freq="D").date
        cal = pd.DataFrame({"_day": idx})
        if not g_hist.empty:
            cal = cal.merge(g_hist, on="_day", how="left")
        cal["count"] = cal["count"].fillna(0).astype(int)

        # stats over the trailing window (already excludes today)
        counts = cal["count"].to_numpy()
        hist_mean = float(np.mean(counts)) if len(counts) else 0.0
        hist_std  = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
        hist_pctl = float(np.quantile(counts, pct)) if len(counts) else 0.0
        hist_active = int((counts > 0).sum())

        base_rows.append({
            keys[0]: a, keys[1]: b, keys[2]: c, "_day": day_today,
            "group_count_today": cnt_today,
            "hist_mean_30d": hist_mean,
            "hist_std_30d": hist_std,
            "hist_pctl_30d": hist_pctl,
            "hist_active_days": hist_active
        })

    base_df = pd.DataFrame(base_rows)

    # --- decide flags ---
    if method == "zscore":
        std0 = (base_df["hist_std_30d"] == 0) & (base_df["group_count_today"] > base_df["hist_mean_30d"])
        stdk = (base_df["hist_std_30d"] > 0) & (
            base_df["group_count_today"] > base_df["hist_mean_30d"] + z_k * base_df["hist_std_30d"]
        )
        flag = (base_df["hist_active_days"] >= min_active_days) & (std0 | stdk)
        thresh_txt = np.where(
            std0,
            (base_df["hist_mean_30d"].round(2)).astype(str) + " (std≈0)",
            (base_df["hist_mean_30d"] + z_k * base_df["hist_std_30d"]).round(2).astype(str)
        )
    else:
        flag = (base_df["hist_active_days"] >= min_active_days) & (base_df["group_count_today"] > base_df["hist_pctl_30d"])
        thresh_txt = base_df["hist_pctl_30d"].round(2).astype(str)

    base_df["vol_spike_flag"] = flag.astype(int)
    base_df["vol_spike_reason"] = [
        f"Txn volume spike: {int(c)} today vs typical {t} for Account='{a}', BU='{b}', Code='{cd}' (last {horizon_days}d)."
        for c, t, a, b, cd in zip(
            base_df["group_count_today"], thresh_txt, base_df[keys[0]], base_df[keys[1]], base_df[keys[2]]
        )
    ]

    # --- attach back to row-level ---
    out = out.merge(
        base_df[[*keys, "_day", "group_count_today", "hist_mean_30d", "hist_std_30d",
                 "hist_pctl_30d", "hist_active_days", "vol_spike_flag", "vol_spike_reason"]],
        on=[*keys, "_day"], how="left"
    ).drop(columns=["_day"])

    # defaults for groups that somehow didn’t get stats
    for c in ["group_count_today","hist_mean_30d","hist_std_30d","hist_pctl_30d","hist_active_days","vol_spike_flag"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    if "vol_spike_reason" in out.columns:
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
