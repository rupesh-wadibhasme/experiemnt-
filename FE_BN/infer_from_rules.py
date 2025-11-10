from datetime import date
from rule_anomalies import (
    load_volume_baseline,
    flag_non_business_days,
    flag_daily_volume_spikes_with_baseline,
)

baseline_df = load_volume_baseline("volume_baseline.pkl")
HOLIDAYS = {date(2025, 1, 26), date(2025, 8, 15)}  # example

df_train = pd.read_excel("bank_history_3yrs.xlsx")

# Non-business day rule
df_train_nb = flag_non_business_days(
    df_train,
    value_date_col="ValueDateKey",
    posting_date_col="PostingDateKey",
    holidays=HOLIDAYS,
)

# Volume spikes using training baseline
df_train_rules = flag_daily_volume_spikes_with_baseline(
    df_train_nb,
    baseline_df=baseline_df,
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
    posting_date_col="PostingDateKey",
    min_history_days=5,
    zscore_threshold=3.0,
)

# Extract anomalies
nonbiz_anoms = df_train_rules[
    (df_train_rules["is_nonbiz_value"] == 1) |
    (df_train_rules["is_nonbiz_post"] == 1)
]

vol_spike_anoms = df_train_rules[df_train_rules["is_volume_spike"] == 1]

nonbiz_anoms.to_csv("nonbiz_anomalies_train.csv", index=False)
vol_spike_anoms.to_csv("volume_spike_anomalies_train.csv", index=False)
