import os
import re
import pandas as pd
from typing import Tuple, Set, Dict, List

# uses your baseline_store helpers
from baseline_store import (
    DailyCountStore,
    flag_spikes_from_store,
    flag_non_business_day_rules,
)

# ---------- column resolver (case/space/underscore-insensitive) ----------

def _norm(s: str) -> str:
    """normalize a column name: lowercase, remove non-alphanumerics."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _resolve_columns(df: pd.DataFrame, want: Dict[str, List[str]]) -> Dict[str, str]:
    """
    want = {
      "BankAccountCode": ["BankAccountCode","bank_account_code","bank account code", ...],
      "BusinessUnitCode": [...],
      "BankTransactionCode": [...],
      "PostingDateKey": [...],
      "ValueDateKey": [...],
    }
    returns mapping of logical -> actual df column names
    """
    norm_map = {_norm(c): c for c in df.columns}
    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for logical, candidates in want.items():
        found = None
        for cand in candidates:
            n = _norm(cand)
            if n in norm_map:
                found = norm_map[n]
                break
        if found is None:
            missing.append(logical)
        else:
            resolved[logical] = found

    if missing:
        # helpful error: show what columns exist
        preview = ", ".join(list(df.columns)[:50])
        raise KeyError(
            f"Could not resolve required column(s): {missing}. "
            f"Available columns (first 50): {preview}\n"
            f"Tip: make sure you passed the engineered pandas DataFrame (e.g., feats_train) "
            f"from FE_BN_5.build_training_matrix_from_excel(...), not X_train."
        )
    return resolved

def _want_map(
    acct="BankAccountCode",
    bu="BusinessUnitCode",
    code="BankTransactionCode",
    post="PostingDateKey",
    val="ValueDateKey",
):
    return {
        "BankAccountCode": [acct, "bank account code", "bank_account_code", "acct_code", "accountcode"],
        "BusinessUnitCode": [bu, "business unit code", "business_unit_code", "bu_code", "businessunit"],
        "BankTransactionCode": [code, "bank transaction code", "bank_transaction_code", "txn_code", "transactioncode"],
        "PostingDateKey": [post, "posting date key", "posting_date_key", "postingdate"],
        "ValueDateKey": [val, "value date key", "value_date_key", "valuedate"],
        "AmountInBankAccountCurrency": ["AmountInBankAccountCurrency", "amount", "amountinbankaccountcurrency"],
        "ts": ["ts", "timestamp", "post_ts", "posting_ts"],
    }

# ------------------ rule-only anomaly runner (robust) --------------------

def run_rule_anomalies_only(
    feats_df: pd.DataFrame,
    out_dir: str = "rule_outputs",
    store_root: str = "artifacts_features/baselines/daily_counts",
    holidays: Set = None,
    method: str = "zscore",        # or "percentile"
    horizon_days: int = 30,
    z_k: float = 3.0,
    pct: float = 0.99,
    min_active_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    AE-independent rule anomaly detection with resilient column resolution:
      (1) Non-business day anomalies (weekend/holiday)
      (2) Daily volume spikes for (BankAccountCode, BusinessUnitCode, BankTransactionCode)

    Saves two CSVs under `out_dir` and returns (nbz_df, vol_df).
    """
    if not isinstance(feats_df, pd.DataFrame):
        raise TypeError(
            "run_rule_anomalies_only expects a pandas DataFrame (e.g., feats_train). "
            "You likely passed X_train (numpy array)."
        )

    os.makedirs(out_dir, exist_ok=True)
    holidays = holidays or set()

    # Resolve columns flexibly
    want = _want_map()
    col = _resolve_columns(feats_df, want)

    acct_col  = col["BankAccountCode"]
    bu_col    = col["BusinessUnitCode"]
    code_col  = col["BankTransactionCode"]
    post_col  = col["PostingDateKey"]
    val_col   = col["ValueDateKey"]
    amt_col   = col.get("AmountInBankAccountCurrency", "AmountInBankAccountCurrency")
    ts_col    = col.get("ts", "ts") if "ts" in col else None

    # ---- (1) Non-business day anomalies ----
    nbz_frame = flag_non_business_day_rules(
        feats_df,
        value_col=val_col,
        posting_col=post_col,
        holidays=holidays,
    )
    nbz_mask = (nbz_frame.get("is_nonbiz_value", 0) == 1) | (nbz_frame.get("is_nonbiz_post", 0) == 1)
    nbz_cols = [
        ts_col, val_col, post_col,
        acct_col, bu_col, code_col,
        amt_col,
        "is_nonbiz_value", "is_nonbiz_post", "nonbiz_reason",
    ]
    nbz_keep = [c for c in nbz_cols if c is not None and c in nbz_frame.columns]
    nbz_df = nbz_frame.loc[nbz_mask, nbz_keep].copy()
    nbz_path = os.path.join(out_dir, "nonbiz_anomalies.csv")
    nbz_df.to_csv(nbz_path, index=False)

    # ---- (2) Volume spike anomalies via DailyCountStore (no leakage) ----
    store = DailyCountStore(root=store_root)

    spk_frame = flag_spikes_from_store(
        feats_df,
        store=store,
        keys=(acct_col, bu_col, code_col),   # pass resolved names
        posting_col=post_col,
        method=method,
        horizon_days=horizon_days,
        z_k=z_k,
        pct=pct,
        min_active_days=min_active_days,
    )
    vol_mask = spk_frame.get("vol_spike_flag", 0) == 1
    vol_cols = [
        ts_col,
        acct_col, bu_col, code_col,
        "group_count_today", "hist_mean_30d", "hist_std_30d",
        "hist_pctl_30d", "hist_active_days", "vol_spike_flag", "vol_spike_reason",
    ]
    vol_keep = [c for c in vol_cols if c is not None and c in spk_frame.columns]
    vol_df = spk_frame.loc[vol_mask, vol_keep].copy()
    vol_path = os.path.join(out_dir, "volume_spike_anomalies.csv")
    vol_df.to_csv(vol_path, index=False)

    # Update store AFTER flagging (prevents look-ahead leakage)
    store.upsert_counts(
        feats_df,
        keys=(acct_col, bu_col, code_col),
        posting_col=post_col,
    )

    print(f"[rules] Saved: {nbz_path} ({len(nbz_df)}) and {vol_path} ({len(vol_df)})")
    return nbz_df, vol_df
