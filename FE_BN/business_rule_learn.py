import pandas as pd
import numpy as np
import pickle
from datetime import date
from typing import Set, List


# ============ Common helpers ============

def _to_datetime(s: pd.Series) -> pd.Series:
    """
    Parse YYYYMMDD-like strings/ints into pandas.Timestamp.
    """
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")


# ============ 1) Baseline: combo frequencies ============

def compute_combo_baseline(
    df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
) -> pd.DataFrame:
    """
    Compute historical frequency of each (account, BU, transaction code) combination.
    Returns a dataframe with:
      [account_col, bu_col, code_col, combo_total_count]
    """
    grp_cols = [account_col, bu_col, code_col]
    base = (
        df
        .groupby(grp_cols, dropna=False)
        .size()
        .reset_index(name="combo_total_count")
    )
    return base


def save_combo_baseline(baseline_df: pd.DataFrame, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(baseline_df, f)


def load_combo_baseline(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        baseline_df = pickle.load(f)
    return baseline_df


# ============ 2) Rule: non-business days (weekend / holidays) ============

def flag_non_business_days(
    df: pd.DataFrame,
    value_col: str = "ValueDateKey",
    posting_col: str = "PostingDateKey",
    holidays: Set[date] = None,
) -> pd.DataFrame:
    """
    Adds:
      - is_nonbiz_value (0/1)
      - is_nonbiz_post  (0/1)
      - nonbiz_reason   (text)
    Flags weekends and provided holidays for value and posting dates.
    """
    holidays = holidays or set()
    out = df.copy()

    # parse dates
    vdt = _to_datetime(out[value_col]) if value_col in out.columns else pd.to_datetime(pd.NaT)
    pdt = _to_datetime(out[posting_col]) if posting_col in out.columns else pd.to_datetime(pd.NaT)

    val_weekend = vdt.dt.dayofweek.isin([5, 6])
    post_weekend = pdt.dt.dayofweek.isin([5, 6])

    val_holiday = vdt.dt.date.isin(holidays) if holidays else pd.Series(False, index=out.index)
    post_holiday = pdt.dt.date.isin(holidays) if holidays else pd.Series(False, index=out.index)

    out["is_nonbiz_value"] = (val_weekend | val_holiday).astype(int)
    out["is_nonbiz_post"]  = (post_weekend | post_holiday).astype(int)

    reasons = []
    for i in range(len(out)):
        rv = ""
        if out.loc[i, "is_nonbiz_post"] == 1:
            why = "weekend" if post_weekend.iloc[i] else "holiday"
            rv = f"Posting date {pdt.iloc[i].date() if pd.notna(pdt.iloc[i]) else 'NA'} is a non-business day ({why})."
        elif out.loc[i, "is_nonbiz_value"] == 1:
            why = "weekend" if val_weekend.iloc[i] else "holiday"
            rv = f"Value date {vdt.iloc[i].date() if pd.notna(vdt.iloc[i]) else 'NA'} is a non-business day ({why})."
        reasons.append(rv)

    out["nonbiz_reason"] = reasons
    return out


# ============ 3) Rule: posting date ≠ value date ============

def flag_posting_value_mismatch(
    df: pd.DataFrame,
    value_col: str = "ValueDateKey",
    posting_col: str = "PostingDateKey",
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
) -> pd.DataFrame:
    """
    Adds:
      - is_posting_value_diff (0/1)
      - posting_diff_reason   (text)
    Flags rows where posting date != value date (and both are valid).
    """
    out = df.copy()

    vdt = _to_datetime(out[value_col]) if value_col in out.columns else pd.to_datetime(pd.NaT)
    pdt = _to_datetime(out[posting_col]) if posting_col in out.columns else pd.to_datetime(pd.NaT)

    mismatch = (vdt.notna()) & (pdt.notna()) & (vdt.dt.date != pdt.dt.date)
    out["is_posting_value_diff"] = mismatch.astype(int)

    reasons = []
    for i in range(len(out)):
        if mismatch.iloc[i]:
            acct = out.loc[i, account_col] if account_col in out.columns else "?"
            bu   = out.loc[i, bu_col]      if bu_col in out.columns else "?"
            code = out.loc[i, code_col]    if code_col in out.columns else "?"
            rv = (
                f"Posting date {pdt.iloc[i].date()} differs from value date {vdt.iloc[i].date()} "
                f"for Account='{acct}', BU='{bu}', TxnCode='{code}'."
            )
        else:
            rv = ""
        reasons.append(rv)

    out["posting_diff_reason"] = reasons
    return out


# ============ 4) Rule: first-time combo transactions ============

def flag_first_combo_transactions(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
) -> pd.DataFrame:
    """
    Uses baseline combo frequencies to flag "first" transactions for each combination.

    baseline_df must have:
      [account_col, bu_col, code_col, combo_total_count]

    Adds:
      - hist_combo_count   (float, NaN if never seen before)
      - is_first_combo_txn (0/1)
      - first_combo_reason (text)
    """
    out = df.copy()
    grp_cols = [account_col, bu_col, code_col]

    if not set(grp_cols).issubset(out.columns):
        missing = [c for c in grp_cols if c not in out.columns]
        raise KeyError(f"Missing combo columns in df: {missing}")

    if not set(grp_cols + ["combo_total_count"]).issubset(baseline_df.columns):
        raise KeyError("baseline_df must have columns "
                       f"{grp_cols + ['combo_total_count']}")

    # join baseline frequencies
    out = out.merge(
        baseline_df[grp_cols + ["combo_total_count"]],
        on=grp_cols,
        how="left",
    )

    # hist count per combo
    hist = out["combo_total_count"]
    # treat NaN as 0 for explanation
    hist_filled = hist.fillna(0)

    # "first" means: only 1 historical occurrence, or none (NaN)
    is_first = (hist_filled <= 1)
    out["hist_combo_count"] = hist_filled
    out["is_first_combo_txn"] = is_first.astype(int)

    reasons = []
    for i in range(len(out)):
        if is_first.iloc[i]:
            acct = out.loc[i, account_col]
            bu   = out.loc[i, bu_col]
            code = out.loc[i, code_col]
            cnt  = hist_filled.iloc[i]
            rv = (
                f"This appears to be the first transaction for combination "
                f"(Account='{acct}', BU='{bu}', TxnCode='{code}') in historical data "
                f"(count={int(cnt)})."
            )
        else:
            rv = ""
        reasons.append(rv)

    out["first_combo_reason"] = reasons
    return out


# ============ 5) Combine reasons into one column ============

def combine_reasons_columns(
    df: pd.DataFrame,
    reason_cols: List[str],
    out_col: str = "anomaly_reason",
) -> pd.DataFrame:
    """
    Combine multiple reason text columns into a single numbered column.

    Example with reason_cols=["nonbiz_reason","posting_diff_reason","first_combo_reason"]:
      "1) <nonbiz> 2) <posting> 3) <first_combo>"

    If only one reason exists, just "1) <that reason>".
    If none, empty string.
    """
    out = df.copy()
    combined = []

    n = len(out)
    # pre-fetch all as strings
    cols_present = {c: out[c].fillna("").astype(str) if c in out.columns
                    else pd.Series([""] * n, index=out.index)
                    for c in reason_cols}

    for i in range(n):
        parts = []
        idx = 1
        for c in reason_cols:
            txt = cols_present[c].iloc[i].strip()
            if txt:
                parts.append(f"{idx}) {txt}")
                idx += 1
        combined.append(" ".join(parts))

    out[out_col] = combined
    return out
