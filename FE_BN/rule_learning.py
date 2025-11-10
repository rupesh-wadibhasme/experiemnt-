
#
# Pure rule-based anomaly detection with train/inference split:
# 1) Non-business day anomalies (weekend / holiday)
# 2) Daily volume spikes for (BankAccountCode, BusinessUnitCode, BankTransactionCode)
#
# Training phase:
#   - compute_volume_baseline(train_df, ...) -> baseline_stats (per combo)
#   - save_volume_baseline(baseline_stats, "volume_baseline.pkl")
#
# Inference phase:
#   - baseline_stats = load_volume_baseline("volume_baseline.pkl")
#   - df_out = flag_daily_volume_spikes_with_baseline(test_df, baseline_stats, ...)
#   - plus optional flag_non_business_days(...)

from __future__ import annotations
from typing import Optional, Tuple, Set
from datetime import date
import pickle

import numpy as np
import pandas as pd


# =========================================================
# Helpers for dates
# =========================================================

def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert typical YYYYMMDD / date-like columns to pandas.Timestamp.
    Works with int, str; invalid/missing -> NaT.
    """
    return pd.to_datetime(series.astype(str), errors="coerce")


def _is_weekend(dt: pd.Series) -> pd.Series:
    """Return boolean Series: True if Saturday (5) or Sunday (6)."""
    return dt.dt.dayofweek.isin([5, 6])


def _is_holiday(dt: pd.Series, holidays: Set[date]) -> pd.Series:
    """Return boolean Series: True if date in given holiday set."""
    if not holidays:
        return pd.Series(False, index=dt.index)
    return dt.dt.date.isin(holidays)


# =========================================================
# 1) Non-business day rule  (no training needed)
# =========================================================

def flag_non_business_days(
    df: pd.DataFrame,
    value_date_col: str = "ValueDateKey",
    posting_date_col: str = "PostingDateKey",
    holidays: Optional[Set[date]] = None,
) -> pd.DataFrame:
    """
    Add flags for non-business days:
      - is_nonbiz_value: 1 if ValueDate is weekend or holiday
      - is_nonbiz_post:  1 if PostingDate is weekend or holiday
      - nonbiz_reason:   human readable string (or empty if not flagged)

    This rule is stateless and can be used directly at inference time.
    """
    holidays = holidays or set()
    out = df.copy()

    vdt = _to_datetime(out[value_date_col]) if value_date_col in out.columns else pd.to_datetime(pd.NaT)
    pdt = _to_datetime(out[posting_date_col]) if posting_date_col in out.columns else pd.to_datetime(pd.NaT)

    val_weekend = _is_weekend(vdt)
    post_weekend = _is_weekend(pdt)
    val_holiday = _is_holiday(vdt, holidays)
    post_holiday = _is_holiday(pdt, holidays)

    out["is_nonbiz_value"] = (val_weekend | val_holiday).astype(int)
    out["is_nonbiz_post"]  = (post_weekend | post_holiday).astype(int)

    reasons = []
    for i in range(len(out)):
        reason = ""
        if post_weekend.iloc[i]:
            reason = f"Posting date {pdt.iloc[i].date()} is a weekend."
        elif post_holiday.iloc[i]:
            reason = f"Posting date {pdt.iloc[i].date()} is a holiday."
        elif val_weekend.iloc[i]:
            reason = f"Value date {vdt.iloc[i].date()} is a weekend."
        elif val_holiday.iloc[i]:
            reason = f"Value date {vdt.iloc[i].date()} is a holiday."
        reasons.append(reason)

    out["nonbiz_reason"] = reasons
    return out


# =========================================================
# 2) Daily volume spike rule — TRAINING phase
# =========================================================

def compute_volume_baseline(
    df_train: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    posting_date_col: str = "PostingDateKey",
) -> pd.DataFrame:
    """
    Compute baseline stats for daily transaction counts per
    (account, BU, code) using a historical TRAINING dataset.

    Returns a DataFrame with columns:
      [account_col, bu_col, code_col,
       mean_daily_count, std_daily_count, active_days]

    This baseline can be saved to pickle and later used for inference.
    """
    if posting_date_col not in df_train.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in training dataframe")

    train = df_train.copy()
    posting_ts = _to_datetime(train[posting_date_col])
    train["_posting_date"] = posting_ts.dt.date

    # Daily counts per (acct, BU, code, date)
    grp_keys = [account_col, bu_col, code_col, "_posting_date"]
    daily = (
        train.groupby(grp_keys, dropna=False)
             .size()
             .reset_index(name="group_count_today")
    )

    # Stats per (acct, BU, code)
    baseline = (
        daily.groupby([account_col, bu_col, code_col], dropna=False)["group_count_today"]
             .agg(
                 mean_daily_count="mean",
                 std_daily_count="std",
                 active_days="size",
             )
             .reset_index()
    )

    # Fill NaNs in std with 0 for convenience
    baseline["std_daily_count"] = baseline["std_daily_count"].fillna(0.0)

    return baseline


def save_volume_baseline(baseline_df: pd.DataFrame, path: str) -> None:
    """Save baseline stats (DataFrame) to pickle."""
    with open(path, "wb") as f:
        pickle.dump(baseline_df, f)


def load_volume_baseline(path: str) -> pd.DataFrame:
    """Load baseline stats (DataFrame) from pickle."""
    with open(path, "rb") as f:
        baseline_df = pickle.load(f)
    return baseline_df


# =========================================================
# 2) Daily volume spike rule — INFERENCE phase
# =========================================================

def flag_daily_volume_spikes_with_baseline(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    posting_date_col: str = "PostingDateKey",
    min_history_days: int = 5,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Use PRECOMPUTED baseline stats (from compute_volume_baseline)
    to flag days where today's count is unusually high.

    Steps:
      1. For incoming df, compute group_count_today per
         (account, BU, code, posting_date).
      2. Join with baseline_df on (account, BU, code) to get:
           - mean_daily_count
           - std_daily_count
           - active_days
      3. For groups with active_days >= min_history_days:
           - if std > 0:
               spike if count_today > mean + zscore_threshold * std
           - if std == 0:
               spike if count_today > mean
      4. Add columns to df:
           - group_count_today
           - mean_daily_count
           - std_daily_count
           - active_days
           - is_volume_spike
           - volume_spike_reason
    """
    if posting_date_col not in df.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in dataframe")

    out = df.copy()

    # derive posting_date as calendar date
    posting_ts = _to_datetime(out[posting_date_col])
    out["_posting_date"] = posting_ts.dt.date

    grp_keys = [account_col, bu_col, code_col, "_posting_date"]
    daily = (
        out.groupby(grp_keys, dropna=False)
           .size()
           .reset_index(name="group_count_today")
    )

    # join with baseline stats
    daily = daily.merge(
        baseline_df,
        on=[account_col, bu_col, code_col],
        how="left",
        validate="m:1"  # ensure baseline is per combination
    )

    # fill missing stats with 0 (means "no history" → can't spike)
    daily["mean_daily_count"] = daily["mean_daily_count"].fillna(0.0)
    daily["std_daily_count"]  = daily["std_daily_count"].fillna(0.0)
    daily["active_days"]      = daily["active_days"].fillna(0).astype(int)

    # spike logic
    std_pos = daily["std_daily_count"] > 0
    has_hist = daily["active_days"] >= min_history_days

    cond1 = (
        has_hist &
        std_pos &
        (daily["group_count_today"] >
         daily["mean_daily_count"] + zscore_threshold * daily["std_daily_count"])
    )

    std_zero = ~std_pos
    cond2 = (
        has_hist &
        std_zero &
        (daily["group_count_today"] > daily["mean_daily_count"])
    )

    daily["is_volume_spike"] = (cond1 | cond2)

    # build human-readable reason
    reasons = []
    for _, row in daily.iterrows():
        if not row["is_volume_spike"]:
            reasons.append("")
            continue
        acct = row[account_col]
        bu   = row[bu_col]
        code = row[code_col]
        cnt_today = int(row["group_count_today"])
        mean_cnt  = float(row["mean_daily_count"])
        std_cnt   = float(row["std_daily_count"])
        d         = row["_posting_date"]
        reasons.append(
            f"On {d}, there were {cnt_today} transactions for "
            f"(Account='{acct}', BU='{bu}', TxnCode='{code}'), "
            f"vs baseline {mean_cnt:.2f}±{std_cnt:.2f} per day."
        )
    daily["volume_spike_reason"] = reasons

    # attach back to row-level df
    out = out.merge(
        daily[[account_col, bu_col, code_col, "_posting_date",
               "group_count_today", "mean_daily_count", "std_daily_count",
               "active_days", "is_volume_spike", "volume_spike_reason"]],
        on=[account_col, bu_col, code_col, "_posting_date"],
        how="left",
    )

    # cleanup/fill
    for c in ["group_count_today", "mean_daily_count", "std_daily_count", "active_days"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    out["is_volume_spike"] = out["is_volume_spike"].fillna(False).astype(int)
    out["volume_spike_reason"] = out["volume_spike_reason"].fillna("")

    out.drop(columns=["_posting_date"], inplace=True)
    return out


# =========================================================
# Optional convenience wrappers
# =========================================================

def apply_rules_with_baseline(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    value_date_col: str = "ValueDateKey",
    posting_date_col: str = "PostingDateKey",
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    holidays: Optional[Set[date]] = None,
    min_history_days: int = 5,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Run:
      - non-business day flags
      - volume spikes using precomputed baseline
    on a single dataframe.
    """
    df1 = flag_non_business_days(
        df,
        value_date_col=value_date_col,
        posting_date_col=posting_date_col,
        holidays=holidays,
    )
    df2 = flag_daily_volume_spikes_with_baseline(
        df1,
        baseline_df=baseline_df,
        account_col=account_col,
        bu_col=bu_col,
        code_col=code_col,
        posting_date_col=posting_date_col,
        min_history_days=min_history_days,
        zscore_threshold=zscore_threshold,
    )
    return df2
