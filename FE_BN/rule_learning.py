# rule_anomalies.py
#
# Rule-based anomaly detection for bank statements:
# 1) Non-business-day flags (weekend / holiday on Value/Posting date)
# 2) Daily volume "extra transaction" flags:
#    - Learn baseline daily counts per (Account, BU, TxnCode) from training data
#    - At inference time, flag ONLY the extra txns above baseline threshold
#    - For each flagged txn, also list ALL txn IDs of that combo+day in `group_txn_ids`

from __future__ import annotations
from typing import Optional, Set
import pickle
import numpy as np
import pandas as pd


# =============================================================================
# Common helpers
# =============================================================================

def _to_datetime(series: pd.Series) -> pd.Series:
    """Safe conversion to pandas.Timestamp; invalid entries -> NaT."""
    return pd.to_datetime(series.astype(str), errors="coerce")


# =============================================================================
# 1) Non-business day rule (weekend / holiday)
# =============================================================================

def flag_non_business_days(
    df: pd.DataFrame,
    value_col: str = "ValueDateKey",
    posting_col: str = "PostingDateKey",
    holidays: Optional[Set] = None,
) -> pd.DataFrame:
    """
    Flag transactions where ValueDate or PostingDate falls on weekend or holiday.

    Adds:
      - is_nonbiz_value (0/1)
      - is_nonbiz_post  (0/1)
      - nonbiz_reason   (str)
    """
    holidays = holidays or set()
    out = df.copy()

    # Convert to datetime
    if value_col in out.columns:
        val_dt = _to_datetime(out[value_col])
    else:
        val_dt = pd.to_datetime(pd.NaT)

    if posting_col in out.columns:
        pst_dt = _to_datetime(out[posting_col])
    else:
        pst_dt = pd.to_datetime(pd.NaT)

    # Weekend flags
    val_weekend = val_dt.dt.dayofweek.isin([5, 6])
    pst_weekend = pst_dt.dt.dayofweek.isin([5, 6])

    # Holiday flags
    if holidays:
        val_holiday = val_dt.dt.date.isin(holidays)
        pst_holiday = pst_dt.dt.date.isin(holidays)
    else:
        val_holiday = pd.Series(False, index=out.index)
        pst_holiday = pd.Series(False, index=out.index)

    out["is_nonbiz_value"] = (val_weekend | val_holiday).astype(int)
    out["is_nonbiz_post"]  = (pst_weekend | pst_holiday).astype(int)

    # Build a simple human-readable reason
    reasons = []
    for i in range(len(out)):
        r = ""
        if bool(pst_weekend.iat[i]) or bool(pst_holiday.iat[i]):
            when = pst_dt.iat[i]
            why = "weekend" if bool(pst_weekend.iat[i]) else "holiday"
            r = f"PostingDate {when.strftime('%Y-%m-%d') if pd.notna(when) else 'NA'} is a non-business day ({why})."
        elif bool(val_weekend.iat[i]) or bool(val_holiday.iat[i]):
            when = val_dt.iat[i]
            why = "weekend" if bool(val_weekend.iat[i]) else "holiday"
            r = f"ValueDate {when.strftime('%Y-%m-%d') if pd.notna(when) else 'NA'} is a non-business day ({why})."
        reasons.append(r)

    out["nonbiz_reason"] = reasons
    return out


# =============================================================================
# 2) Daily volume baseline (TRAINING)
# =============================================================================

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

    - mean_daily_count: average daily count for that combo
    - std_daily_count:  std dev of daily count for that combo
    - active_days:      number of distinct days with >=1 transaction for that combo
    """
    if posting_date_col not in df_train.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in training dataframe")

    train = df_train.copy()
    posting_ts = _to_datetime(train[posting_date_col])
    train["_posting_date"] = posting_ts.dt.date

    # daily counts per (acct, BU, code, date)
    grp_keys = [account_col, bu_col, code_col, "_posting_date"]
    daily = (
        train.groupby(grp_keys, dropna=False)
             .size()
             .reset_index(name="group_count_today")
    )

    # stats per (acct, BU, code)
    baseline = (
        daily.groupby([account_col, bu_col, code_col], dropna=False)["group_count_today"]
             .agg(
                 mean_daily_count="mean",
                 std_daily_count="std",
                 active_days="size",
             )
             .reset_index()
    )
    baseline["std_daily_count"] = baseline["std_daily_count"].fillna(0.0)

    return baseline


def save_volume_baseline(baseline_df: pd.DataFrame, path: str) -> None:
    """Save baseline stats to pickle."""
    with open(path, "wb") as f:
        pickle.dump(baseline_df, f)


def load_volume_baseline(path: str) -> pd.DataFrame:
    """Load baseline stats from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# 3) Daily volume "extra transactions" rule (INFERENCE)
# =============================================================================

def flag_extra_volume_txns_with_baseline(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    posting_date_col: str = "PostingDateKey",
    txn_id_col: str = "BankTransactionId",   # which column holds transaction IDs
    min_history_days: int = 5,
    zscore_threshold: float = 3.0,
    sort_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Use PRECOMPUTED baseline stats to flag only the "extra" transactions
    on high-volume days.

    Logic:
      1. For the input df, compute daily count per (account, BU, code, posting_date)
         => group_count_today.

      2. Join with baseline_df (per (account, BU, code)) to get:
         mean_daily_count, std_daily_count, active_days.

      3. For each group with sufficient history (active_days >= min_history_days):
           - if std > 0:
                 threshold = mean + zscore_threshold * std
             else:
                 threshold = mean
           - Convert threshold to integer:
                 threshold_int = max(1, round(threshold))
           - extra = max(0, group_count_today - threshold_int)

      4. For groups with extra > 0:
           - collect all rows in df for that (acct, BU, code, date)
           - sort them by `sort_col` if given, otherwise by original index
           - mark ONLY the last `extra` rows as is_volume_spike_txn = 1
           - in those flagged rows, also put ALL txn IDs of that group-day
             into `group_txn_ids` (comma-separated string)

      5. Adds columns to df:
           - group_count_today
           - mean_daily_count
           - std_daily_count
           - active_days
           - is_volume_spike_txn   (0/1 per row)
           - volume_spike_reason   (text for flagged rows, "" otherwise)
           - group_txn_ids         (on flagged rows: all txn IDs in that group-day)
    """
    if posting_date_col not in df.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in dataframe")

    out = df.copy()

    # derive posting_date
    posting_ts = _to_datetime(out[posting_date_col])
    out["_posting_date"] = posting_ts.dt.date

    grp_keys = [account_col, bu_col, code_col, "_posting_date"]

    # daily counts on THIS dataset
    daily = (
        out.groupby(grp_keys, dropna=False)
           .size()
           .reset_index(name="group_count_today")
    )

    # join with baseline
    daily = daily.merge(
        baseline_df,
        on=[account_col, bu_col, code_col],
        how="left",
        validate="m:1",   # each group-day maps to one baseline row
    )

    # fill missing baseline (no history)
    daily["mean_daily_count"] = daily["mean_daily_count"].fillna(0.0)
    daily["std_daily_count"]  = daily["std_daily_count"].fillna(0.0)
    daily["active_days"]      = daily["active_days"].fillna(0).astype(int)

    # compute thresholds & extras
    mean_ = daily["mean_daily_count"].values
    std_  = daily["std_daily_count"].values
    cnt   = daily["group_count_today"].values
    act_days = daily["active_days"].values

    has_hist = act_days >= min_history_days
    std_pos = std_ > 0

    # numeric threshold per group-day
    thresh = np.where(
        std_pos,
        mean_ + zscore_threshold * std_,
        mean_,
    )
    # integer threshold (at least 1)
    thresh_int = np.maximum(1, np.round(thresh).astype(int))

    extra = np.where(
        has_hist,
        np.maximum(0, cnt - thresh_int),
        0,
    )

    daily["threshold_int"] = thresh_int
    daily["extra_txn"] = extra

    # prepare output columns on row-level df
    out["group_count_today"] = 0
    out["mean_daily_count"] = 0.0
    out["std_daily_count"] = 0.0
    out["active_days"] = 0
    out["is_volume_spike_txn"] = 0
    out["volume_spike_reason"] = ""
    out["group_txn_ids"] = ""      # NEW: all txn IDs in that group-day (for flagged rows)

    # per (acct, BU, code, date) with extra > 0
    for _, row in daily.iterrows():
        e = int(row["extra_txn"])
        if e <= 0:
            continue

        acct = row[account_col]
        bu   = row[bu_col]
        code = row[code_col]
        d    = row["_posting_date"]

        # mask for this group-day
        mask = (
            (out[account_col] == acct) &
            (out[bu_col] == bu) &
            (out[code_col] == code) &
            (out["_posting_date"] == d)
        )
        idx = out.index[mask]
        if len(idx) == 0:
            continue

        # attach stats to all rows for this group-day
        out.loc[mask, "group_count_today"] = int(row["group_count_today"])
        out.loc[mask, "mean_daily_count"]  = float(row["mean_daily_count"])
        out.loc[mask, "std_daily_count"]   = float(row["std_daily_count"])
        out.loc[mask, "active_days"]       = int(row["active_days"])

        # all txn IDs for this group-day (for context)
        if txn_id_col in out.columns:
            group_ids = out.loc[idx, txn_id_col].astype(str).tolist()
            group_ids_str = ",".join(group_ids)
        else:
            group_ids_str = ""

        # decide which specific rows are "extra"
        if sort_col is not None and sort_col in out.columns:
            idx_sorted = out.loc[idx].sort_values(sort_col).index
        else:
            # fall back to original order
            idx_sorted = idx.sort_values()

        # take last e rows as extra
        flag_idx = idx_sorted[-e:]

        # human-readable reason
        reason_text = (
            f"This transaction is one of {e} extra txns above the typical "
            f"{int(row['threshold_int'])} per day for "
            f"(Account='{acct}', BU='{bu}', TxnCode='{code}'). "
            f"Today had {int(row['group_count_today'])} txns; "
            f"baseline≈{row['mean_daily_count']:.2f}±{row['std_daily_count']:.2f}."
        )

        out.loc[flag_idx, "is_volume_spike_txn"] = 1
        out.loc[flag_idx, "volume_spike_reason"] = reason_text
        if group_ids_str:
            out.loc[flag_idx, "group_txn_ids"] = group_ids_str

    # clean up helper
    out.drop(columns=["_posting_date"], inplace=True)
    return out


# =============================================================================
# Example usage (commented out)
# =============================================================================
if __name__ == "__main__":
    # Example skeleton – adapt paths / column names as needed.
    # import pandas as pd

    # # 1) TRAINING: learn baseline from history
    # train_df = pd.read_excel("bank_history_3yrs.xlsx")
    # baseline_df = compute_volume_baseline(train_df)
    # save_volume_baseline(baseline_df, "volume_baseline.pkl")

    # # 2) INFERENCE: apply rules on some dataset (could be same training set, or a new daily file)
    # baseline_df = load_volume_baseline("volume_baseline.pkl")
    # df = pd.read_excel("today_statement.xlsx")

    # # Non-business-day flags
    # HOLIDAYS = set()  # e.g. {date(2025,1,26), date(2025,8,15)}
    # df_nb = flag_non_business_days(df, holidays=HOLIDAYS)

    # # Volume extra txns
    # df_vol = flag_extra_volume_txns_with_baseline(
    #     df_nb,
    #     baseline_df,
    #     account_col="BankAccountCode",
    #     bu_col="BusinessUnitCode",
    #     code_col="BankTransactionCode",
    #     posting_date_col="PostingDateKey",
    #     txn_id_col="BankTransactionId",
    #     min_history_days=5,
    #     zscore_threshold=3.0,
    #     sort_col="PostingDateKey",   # or a timestamp col if you have one
    # )
    #
    # anomalies = df_vol[
    #     (df_vol["is_nonbiz_value"] == 1) |
    #     (df_vol["is_nonbiz_post"] == 1)  |
    #     (df_vol["is_volume_spike_txn"] == 1)
    # ]
    # anomalies.to_csv("rule_anomalies_output.csv", index=False)
    pass
