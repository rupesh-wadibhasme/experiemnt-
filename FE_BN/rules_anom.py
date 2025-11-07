import os, re
import pandas as pd
from typing import Tuple, Set, Dict, List, Optional
from difflib import get_close_matches

from baseline_store import (
    DailyCountStore,
    flag_spikes_from_store,
    flag_non_business_day_rules,
)

# ---------- robust column resolver (can be bypassed with explicit args) ----------

def _norm(s: str) -> str:
    """lowercase + drop non-alphanumerics"""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _find_best(df_cols: List[str], want_norm: str) -> Optional[str]:
    """best-effort: exact-substring match first, then fuzzy pick"""
    # try substring over normalized names
    norm_map = {_norm(c): c for c in df_cols}
    for n, orig in norm_map.items():
        if want_norm in n:
            return orig
    # fuzzy
    close = get_close_matches(want_norm, list(norm_map.keys()), n=1, cutoff=0.6)
    if close:
        return norm_map[close[0]]
    return None

def _resolve_or_fail(
    df: pd.DataFrame,
    key_cols: Optional[Tuple[str, str, str]],
    date_cols: Optional[Tuple[str, str]],  # (posting, value)
    amount_col: Optional[str],
    ts_col: Optional[str],
    debug: bool = True,
) -> Dict[str, str]:
    """
    Returns a dict of resolved names:
      acct, bu, code, posting, value, amount, ts
    You can pass any as explicit to bypass auto-resolution.
    """
    cols = list(df.columns)

    def pick(default_norm: str, explicit: Optional[str]) -> str:
        if explicit:
            if explicit in df.columns:
                return explicit
            raise KeyError(f"Explicit column '{explicit}' not found. Available: {cols[:50]}")
        hit = _find_best(cols, _norm(default_norm))
        if hit is None:
            raise KeyError(
                f"Could not resolve a column like '{default_norm}'. "
                f"Available (first 50): {cols[:50]}\n"
                "Tip: pass the exact names via key_cols/date_cols/amount_col/ts_col."
            )
        return hit

    acct = pick("BankAccountCode", key_cols[0] if key_cols else None)
    bu   = pick("BusinessUnitCode", key_cols[1] if key_cols else None)
    code = pick("BankTransactionCode", key_cols[2] if key_cols else None)
    posting = pick("PostingDateKey", date_cols[0] if date_cols else None)
    value   = pick("ValueDateKey",   date_cols[1] if date_cols else None)
    amount  = pick("AmountInBankAccountCurrency", amount_col)
    tsname  = pick("ts", ts_col) if (ts_col or "ts" in df.columns) else None

    resolved = dict(
        acct=acct, bu=bu, code=code,
        posting=posting, value=value,
        amount=amount, ts=tsname
    )
    if debug:
        print("[resolver] using columns ->",
              {k: v for k, v in resolved.items() if v is not None})
    return resolved

# ------------------ rule-only anomaly runner (explicit or auto) --------------------

def run_rule_anomalies_only(
    feats_df: pd.DataFrame,
    out_dir: str = "rule_outputs",
    store_root: str = "artifacts_features/baselines/daily_counts",
    holidays: Set = None,
    # OPTIONAL explicit overrides to avoid any guessing:
    key_cols: Optional[Tuple[str, str, str]] = None,    # (acct, bu, code)
    date_cols: Optional[Tuple[str, str]] = None,        # (posting, value)
    amount_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    # spike method config
    method: str = "zscore",        # or "percentile"
    horizon_days: int = 30,
    z_k: float = 3.0,
    pct: float = 0.99,
    min_active_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    AE-independent rule anomaly detection:
      (1) Non-business day anomalies (weekend/holiday)
      (2) Daily volume spikes for (acct,bu,code)
    """
    if not isinstance(feats_df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame (e.g., feats_train), not a numpy array.")

    os.makedirs(out_dir, exist_ok=True)
    holidays = holidays or set()

    # Resolve columns (or use explicit)
    col = _resolve_or_fail(
        feats_df,
        key_cols=key_cols,
        date_cols=date_cols,
        amount_col=amount_col,
        ts_col=ts_col,
        debug=True,
    )
    acct_col, bu_col, code_col = col["acct"], col["bu"], col["code"]
    posting_col, value_col     = col["posting"], col["value"]
    amt_col, tsname            = col["amount"], col["ts"]

    # ---- (1) Non-business day anomalies ----
    nbz_frame = flag_non_business_day_rules(
        feats_df,
        value_col=value_col,
        posting_col=posting_col,
        holidays=holidays,
    )
    nbz_mask = (nbz_frame.get("is_nonbiz_value", 0) == 1) | (nbz_frame.get("is_nonbiz_post", 0) == 1)
    nbz_cols = [
        c for c in [tsname, value_col, posting_col,
                    acct_col, bu_col, code_col,
                    amt_col, "is_nonbiz_value", "is_nonbiz_post", "nonbiz_reason"]
        if c in nbz_frame.columns
    ]
    nbz_df = nbz_frame.loc[nbz_mask, nbz_cols].copy()
    nbz_path = os.path.join(out_dir, "nonbiz_anomalies.csv")
    nbz_df.to_csv(nbz_path, index=False)

    # ---- (2) Volume spike anomalies via DailyCountStore (no leakage) ----
    store = DailyCountStore(root=store_root)
    spk_frame = flag_spikes_from_store(
        feats_df,
        store=store,
        keys=(acct_col, bu_col, code_col),
        posting_col=posting_col,
        method=method,
        horizon_days=horizon_days,
        z_k=z_k,
        pct=pct,
        min_active_days=min_active_days,
    )
    vol_mask = spk_frame.get("vol_spike_flag", 0) == 1
    vol_cols = [
        c for c in [tsname,
                    acct_col, bu_col, code_col,
                    "group_count_today", "hist_mean_30d", "hist_std_30d",
                    "hist_pctl_30d", "hist_active_days", "vol_spike_flag", "vol_spike_reason"]
        if c in spk_frame.columns
    ]
    vol_df = spk_frame.loc[vol_mask, vol_cols].copy()
    vol_path = os.path.join(out_dir, "volume_spike_anomalies.csv")
    vol_df.to_csv(vol_path, index=False)

    # Update store AFTER flagging (prevents look-ahead leakage)
    store.upsert_counts(
        feats_df,
        keys=(acct_col, bu_col, code_col),
        posting_col=posting_col,
    )

    print(f"[rules] nonbiz={len(nbz_df)}  vol_spike={len(vol_df)} -> {out_dir}")
    return nbz_df, vol_df



# After FE_BN_5 build:
X_train, feats_train, feat_names = build_training_matrix_from_excel("DL_raw_files.xlsx")

nbz_df, vol_df = run_rule_anomalies_only(
    feats_df=feats_train,
    out_dir="rule_outputs",
    store_root="artifacts_features/baselines/daily_counts",
    # tell it exactly which columns to use:
    key_cols=("BankAccountCode", "BusinessUnitCode", "BankTransactionCode"),
    date_cols=("PostingDateKey", "ValueDateKey"),
    amount_col="AmountInBankAccountCurrency",
    ts_col="ts",            # if present; otherwise omit
    method="zscore",
    horizon_days=30,
    z_k=3.0,
    pct=0.99,
    min_active_days=7,
)
