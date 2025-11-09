# rule_anomalies.py
#
# Pure rule-based anomaly detection:
# 1) Value / Posting date on weekend or holiday
# 2) Daily volume spike for (BankAccountCode, BusinessUnitCode, BankTransactionCode)

from __future__ import annotations
from typing import Optional, Tuple, Set
from datetime import date
import numpy as np
import pandas as pd


# ---------- helpers for dates ----------

def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert typical YYYYMMDD-like columns to pandas.Timestamp.
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


# ---------- 1) Non-business day rule ----------

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

    Parameters
    ----------
    df : pd.DataFrame
        Input transactions.
    value_date_col : str
        Column with value date (YYYYMMDD / date-like).
    posting_date_col : str
        Column with posting date (YYYYMMDD / date-like).
    holidays : set of datetime.date
        Optional list/set of holiday dates.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with added columns.
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


# ---------- 2) Daily volume spike rule ----------

def flag_daily_volume_spikes(
    df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    posting_date_col: str = "PostingDateKey",
    min_history_days: int = 5,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect days where the number of transactions for a given
    (account, BU, transaction code) is much higher than usual.

    Logic:
      1. Convert posting_date_col to a calendar date (no time).
      2. For each (account, BU, code, date) compute group_count_today.
      3. For each (account, BU, code) compute:
           - mean_daily_count over all days in df
           - std_daily_count
           - active_days (how many days had at least one txn)
      4. A day is a spike if:
           - active_days >= min_history_days, AND
           - group_count_today > mean_daily_count + zscore_threshold * std_daily_count
         Special case: if std=0 and group_count_today > mean_daily_count,
         and active_days >= min_history_days, we also mark as spike.

    Adds columns to the returned DataFrame:
      - group_count_today
      - mean_daily_count
      - std_daily_count
      - active_days
      - is_volume_spike (0/1)
      - volume_spike_reason (string)
    """
    out = df.copy()
    if posting_date_col not in out.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in df")

    # Derive plain calendar date from posting date col
    posting_ts = _to_datetime(out[posting_date_col])
    out["_posting_date"] = posting_ts.dt.date

    # Group by (account, BU, code, posting_date) to get today's count
    grp_keys = [account_col, bu_col, code_col, "_posting_date"]
    daily = (
        out.groupby(grp_keys, dropna=False)
           .size()
           .reset_index(name="group_count_today")
    )

    # Compute stats per (account, BU, code)
    stats = (
        daily.groupby([account_col, bu_col, code_col], dropna=False)["group_count_today"]
             .agg(
                 mean_daily_count="mean",
                 std_daily_count="std",
                 active_days="size"
             )
             .reset_index()
    )

    # Join stats back to daily
    daily = daily.merge(
        stats,
        on=[account_col, bu_col, code_col],
        how="left"
    )

    # Decide spikes
    daily["is_volume_spike"] = False

    # Case 1: std > 0
    std_pos = daily["std_daily_count"].fillna(0) > 0
    cond1 = (
        (daily["active_days"] >= min_history_days) &
        std_pos &
        (daily["group_count_today"] >
         daily["mean_daily_count"] + zscore_threshold * daily["std_daily_count"].fillna(0))
    )

    # Case 2: std == 0 (all days had the same count historically)
    std_zero = ~std_pos
    cond2 = (
        (daily["active_days"] >= min_history_days) &
        std_zero &
        (daily["group_count_today"] > daily["mean_daily_count"])
    )

    daily.loc[cond1 | cond2, "is_volume_spike"] = True

    # Build text reason
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
        std_cnt   = float(row["std_daily_count"]) if pd.notna(row["std_daily_count"]) else 0.0
        d        = row["_posting_date"]

        reasons.append(
            f"On {d}, there were {cnt_today} transactions for "
            f"(Account='{acct}', BU='{bu}', TxnCode='{code}'), "
            f"vs typical {mean_cnt:.2f}±{std_cnt:.2f} per day."
        )

    daily["volume_spike_reason"] = reasons

    # Attach back to row-level dataframe
    out = out.merge(
        daily[[account_col, bu_col, code_col, "_posting_date",
               "group_count_today", "mean_daily_count", "std_daily_count",
               "active_days", "is_volume_spike", "volume_spike_reason"]],
        on=[account_col, bu_col, code_col, "_posting_date"],
        how="left"
    )

    # Cleanup / fill
    for c in ["group_count_today", "mean_daily_count", "std_daily_count", "active_days"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    out["is_volume_spike"] = out["is_volume_spike"].fillna(False).astype(int)
    out["volume_spike_reason"] = out["volume_spike_reason"].fillna("")

    out.drop(columns=["_posting_date"], inplace=True)
    return out


# ---------- convenience wrapper to run both ----------

def apply_simple_rules(
    df: pd.DataFrame,
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
    Run both rule sets on a dataframe and return one enriched dataframe:
      - non-business day flags/reasons
      - daily volume spike flags/reasons
    """
    df1 = flag_non_business_days(
        df,
        value_date_col=value_date_col,
        posting_date_col=posting_date_col,
        holidays=holidays,
    )
    df2 = flag_daily_volume_spikes(
        df1,
        account_col=account_col,
        bu_col=bu_col,
        code_col=code_col,
        posting_date_col=posting_date_col,
        min_history_days=min_history_days,
        zscore_threshold=zscore_threshold,
    )
    return df2
