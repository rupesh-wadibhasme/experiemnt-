from bank_features_mvp_xls import (
    fit_featureizer_from_excel, transform_excel_batch, update_baselines_with_excel
)

# 1) One-time (or on retrain): fit from your 3-year Excel
fit_featureizer_from_excel("bank_history_3yrs.xlsx")

# 2) Daily scoring:
X_scaled, engineered_df, feature_names = transform_excel_batch("bank_batch_2025-10-28.xlsx")
# -> feed X_scaled to your autoencoder

# 3) After successful scoring, keep baselines fresh:
update_baselines_with_excel("bank_batch_2025-10-28.xlsx")
