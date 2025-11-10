import pandas as pd
from datetime import date

from rule_anomalies import (
    compute_volume_baseline,
    save_volume_baseline,
)

# ---- paths ----
TRAIN_XLSX    = "DL_raw_files.xlsx"      # your training file
BASELINE_PKL  = "volume_baseline.pkl"    # where we store baseline stats

# ---- load training data (raw) ----
train_df = pd.read_excel(TRAIN_XLSX, sheet_name=0)

# ---- compute baseline stats per (Account, BU, TxnCode) ----
baseline_df = compute_volume_baseline(
    train_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
    posting_date_col="PostingDateKey",
)

# ---- save baseline to pickle ----
save_volume_baseline(baseline_df, BASELINE_PKL)

print(f"Baseline saved to {BASELINE_PKL} with {len(baseline_df)} groups.")


# -----Inference -----------


import pandas as pd
from datetime import date

from rule_anomalies import (
    load_volume_baseline,
    flag_non_business_days,
    flag_extra_volume_txns_with_baseline,
)

# ---- paths ----
TRAIN_XLSX    = "DL_raw_files.xlsx"          # for now we reuse training file as test
BASELINE_PKL  = "volume_baseline.pkl"
ANOMALY_CSV   = "rule_anomalies_from_train.csv"

# ---- load baseline ----
baseline_df = load_volume_baseline(BASELINE_PKL)

# ---- load data to score (for now: training data itself) ----
df_raw = pd.read_excel(TRAIN_XLSX, sheet_name=0)

# ---- (1) Non-business-day rule ----
# add your holidays here if you have them
HOLIDAYS = {
    # example:
    # date(2025, 1, 26),
    # date(2025, 8, 15),
}
df_nb = flag_non_business_days(
    df_raw,
    value_col="ValueDateKey",
    posting_col="PostingDateKey",
    holidays=HOLIDAYS,
)

# ---- (2) Daily volume extra-transactions rule ----
df_vol = flag_extra_volume_txns_with_baseline(
    df_nb,
    baseline_df=baseline_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
    posting_date_col="PostingDateKey",
    txn_id_col="BankTransactionId",   # must exist in your file
    min_history_days=5,               # need at least 5 days of history to trust baseline
    zscore_threshold=3.0,             # how aggressive the spike definition is
    sort_col="PostingDateKey",        # or a timestamp column if you have one
)

# ---- combine both rule types into a single anomaly view ----
anomalies = df_vol[
    (df_vol["is_nonbiz_value"] == 1) |
    (df_vol["is_nonbiz_post"] == 1)  |
    (df_vol["is_volume_spike_txn"] == 1)
].copy()

# OPTIONAL: keep only the most relevant columns
cols_to_keep = [
    # core IDs
    "BankTransactionId",
    "BankAccountCode",
    "BusinessUnitCode",
    "BankTransactionCode",
    "AmountInBankAccountCurrency",
    "ValueDateKey",
    "PostingDateKey",

    # non-business-day signals
    "is_nonbiz_value",
    "is_nonbiz_post",
    "nonbiz_reason",

    # volume spike signals
    "group_count_today",
    "mean_daily_count",
    "std_daily_count",
    "active_days",
    "is_volume_spike_txn",
    "volume_spike_reason",
    "group_txn_ids",   # ← all txn IDs for that combo+day
]
cols_to_keep = [c for c in cols_to_keep if c in anomalies.columns]
anomalies = anomalies[cols_to_keep]

# ---- save anomalies ----
anomalies.to_csv(ANOMALY_CSV, index=False)
print(f"Saved {len(anomalies)} rule-based anomalies to {ANOMALY_CSV}")
