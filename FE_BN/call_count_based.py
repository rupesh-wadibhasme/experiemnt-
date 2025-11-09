import pandas as pd
from datetime import date
from rule_anomalies import apply_simple_rules

df = pd.read_excel("bank_history_3yrs.xlsx")  # or read_csv

HOLIDAYS = {
    date(2025, 1, 26),
    date(2025, 8, 15),
    # add more...
}

df_rules = apply_simple_rules(
    df,
    value_date_col="ValueDateKey",
    posting_date_col="PostingDateKey",
    account_col="BankAccountCode",
    bu_col="BusinessUnitCode",
    code_col="BankTransactionCode",
    holidays=HOLIDAYS,
    min_history_days=5,    # minimum number of days before we trust the volume baseline
    zscore_threshold=3.0,  # 3σ spike
)

# Non-business-day anomalies:
nbz = df_rules[(df_rules["is_nonbiz_value"] == 1) | (df_rules["is_nonbiz_post"] == 1)]
nbz.to_csv("non_business_day_anomalies.csv", index=False)

# Volume spike anomalies:
spikes = df_rules[df_rules["is_volume_spike"] == 1]
spikes.to_csv("volume_spike_anomalies.csv", index=False)
