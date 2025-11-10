import pandas as pd
from rule_anomalies import compute_volume_baseline, save_volume_baseline

train_df = pd.read_excel("bank_history_3yrs.xlsx")

baseline_df = compute_volume_baseline(
    train_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
    posting_date_col="PostingDateKey",
)

save_volume_baseline(baseline_df, "volume_baseline.pkl")
