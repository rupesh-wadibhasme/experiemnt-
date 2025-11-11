import pandas as pd

from rule_anomalies import (
    compute_combo_baseline,
    save_combo_baseline,
)

# ---- paths ----
TRAIN_XLSX    = "DL_raw_files.xlsx"      # your training file
BASELINE_PKL  = "combo_baseline.pkl"     # new name to reflect meaning (optional)

# ---- load training data (raw) ----
train_df = pd.read_excel(TRAIN_XLSX, sheet_name=0)

# ---- compute baseline stats per (Account, BU, TxnCode) ----
baseline_df = compute_combo_baseline(
    train_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
)

# ---- save baseline to pickle ----
save_combo_baseline(baseline_df, BASELINE_PKL)

print(f"Combo baseline saved to {BASELINE_PKL} with {len(baseline_df)} combinations.")

# Inference ----------------

import pandas as pd
from datetime import date

from rule_anomalies import (
    load_combo_baseline,
    flag_non_business_days,
    flag_posting_value_mismatch,
    flag_first_combo_transactions,
    combine_reasons_columns,
)

# ---- paths ----
TRAIN_XLSX    = "DL_raw_files.xlsx"           # for now we reuse training file as test
BASELINE_PKL  = "combo_baseline.pkl"          # must match what you used above
ANOMALY_CSV   = "rule_anomalies_from_train.csv"

# ---- load baseline ----
baseline_df = load_combo_baseline(BASELINE_PKL)

# ---- load data to score (for now: training data itself) ----
df_raw = pd.read_excel(TRAIN_XLSX, sheet_name=0)

# ---- (1) Non-business-day rule ----
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

# ---- (2) Posting date != value date rule ----
df_mis = flag_posting_value_mismatch(
    df_nb,
    value_col="ValueDateKey",
    posting_col="PostingDateKey",
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
)

# ---- (3) First-transaction combo rule ----
df_fc = flag_first_combo_transactions(
    df_mis,
    baseline_df=baseline_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
)

# ---- (4) Combine all reasons into a single anomaly_reason column ----
df_fc = combine_reasons_columns(
    df_fc,
    reason_cols=["nonbiz_reason", "posting_diff_reason", "first_combo_reason"],
    out_col="anomaly_reason",
)

# ---- (5) Build anomaly view (any rule fired) ----
anomalies = df_fc[
    (df_fc.get("is_nonbiz_value", 0) == 1) |
    (df_fc.get("is_nonbiz_post", 0) == 1)  |
    (df_fc.get("is_posting_value_diff", 0) == 1) |
    (df_fc.get("is_first_combo_txn", 0) == 1)
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

    # rule flags
    "is_nonbiz_value",
    "is_nonbiz_post",
    "is_posting_value_diff",
    "is_first_combo_txn",

    # helpful stats
    "hist_combo_count",

    # unified human reason
    "anomaly_reason",
]
cols_to_keep = [c for c in cols_to_keep if c in anomalies.columns]
anomalies = anomalies[cols_to_keep]

# ---- save anomalies ----
anomalies.to_csv(ANOMALY_CSV, index=False)
print(f"Saved {len(anomalies)} rule-based anomalies to {ANOMALY_CSV}")
