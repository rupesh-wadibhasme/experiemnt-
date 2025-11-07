import os
import pandas as pd
from typing import Tuple, Set

# Uses your file
from baseline_store import (
    DailyCountStore,
    flag_spikes_from_store,       # expects raw cols (not hashed), uses PostingDateKey internally
    flag_non_business_day_rules,  # expects ValueDateKey / PostingDateKey
)

def _ensure_cols(df: pd.DataFrame, cols: list[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{ctx}] Missing required columns: {missing}. "
            f"Make sure you pass the *engineered* dataframe from FE_BN_5 "
            f"that still contains raw columns like BankAccountCode/BusinessUnitCode/BankTransactionCode/ValueDateKey/PostingDateKey."
        )

def run_rule_anomalies_only(
    feats_df: pd.DataFrame,
    out_dir: str = "rule_outputs",
    store_root: str = "artifacts_features/baselines/daily_counts",
    holidays: Set = None,
    keys: Tuple[str, str, str] = ("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
    posting_col: str = "PostingDateKey",
    value_col: str = "ValueDateKey",
    method: str = "zscore",        # or "percentile"
    horizon_days: int = 30,
    z_k: float = 3.0,
    pct: float = 0.99,
    min_active_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    AE-independent rule anomaly detection:
      (1) Non-business day anomalies (weekend/holiday) on ValueDate/PostingDate
      (2) Volume spike anomalies per-day for (Account, BU, Code) combinations, NO leakage via DailyCountStore

    Saves two CSVs under `out_dir` and returns (nbz_df, vol_df).
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Validate columns expected by the two rules ----
    _ensure_cols(feats_df, [value_col, posting_col], "non-business-day rule")
    _ensure_cols(feats_df, list(keys) + [posting_col], "volume-spike rule")

    # ---- (1) Non-business day rule ----
    nbz_frame = flag_non_business_day_rules(
        feats_df,
        value_col=value_col,
        posting_col=posting_col,
        holidays=holidays or set(),
    )
    nbz_mask = (nbz_frame.get("is_nonbiz_value", 0) == 1) | (nbz_frame.get("is_nonbiz_post", 0) == 1)
    nbz_cols = [
        "ts", value_col, posting_col,
        "BankAccountCode", "BusinessUnitCode", "BankTransactionCode",
        "AmountInBankAccountCurrency",
        "is_nonbiz_value", "is_nonbiz_post", "nonbiz_reason",
    ]
    nbz_keep = [c for c in nbz_cols if c in nbz_frame.columns]
    nbz_df = nbz_frame.loc[nbz_mask, nbz_keep].copy()
    nbz_path = os.path.join(out_dir, "nonbiz_anomalies.csv")
    nbz_df.to_csv(nbz_path, index=False)

    # ---- (2) Volume spike rule using DailyCountStore (no leakage) ----
    # IMPORTANT: flag first, update store after
    store = DailyCountStore(root=store_root)

    spk_frame = flag_spikes_from_store(
        feats_df,
        store=store,
        keys=keys,
        posting_col=posting_col,
        method=method,
        horizon_days=horizon_days,
        z_k=z_k,
        pct=pct,
        min_active_days=min_active_days,
    )
    vol_mask = spk_frame.get("vol_spike_flag", 0) == 1
    vol_cols = [
        "ts",
        "BankAccountCode", "BusinessUnitCode", "BankTransactionCode",
        "group_count_today", "hist_mean_30d", "hist_std_30d",
        "hist_pctl_30d", "hist_active_days",
        "vol_spike_flag", "vol_spike_reason",
    ]
    vol_keep = [c for c in vol_cols if c in spk_frame.columns]
    vol_df = spk_frame.loc[vol_mask, vol_keep].copy()
    vol_path = os.path.join(out_dir, "volume_spike_anomalies.csv")
    vol_df.to_csv(vol_path, index=False)

    # ---- Update store AFTER flagging (prevents look-ahead leakage) ----
    store.upsert_counts(
        feats_df,
        keys=keys,
        posting_col=posting_col,
    )

    print(f"[rules] Saved: {nbz_path} ({len(nbz_df)}) and {vol_path} ({len(vol_df)})")
    return nbz_df, vol_df


# # Build features as you already do:
# X_train, feats_train, feat_names = build_training_matrix_from_excel("DL_raw_files.xlsx")

# Now run the rule-only anomalies (no AE involved):
nbz_df, vol_df = run_rule_anomalies_only(
    feats_df=feats_train,
    out_dir="rule_outputs",
    store_root="artifacts_features/baselines/daily_counts",
    holidays=set(),  # or {date(2025, 1, 26), date(2025, 8, 15)} etc.
    keys=("BankAccountCode","BusinessUnitCode","BankTransactionCode"),
    posting_col="PostingDateKey",
    value_col="ValueDateKey",
    method="zscore",           # or "percentile"
    horizon_days=30,
    z_k=3.0,
    pct=0.99,
    min_active_days=7,
)
