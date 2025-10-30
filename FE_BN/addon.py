def make_reason_from_feature(feat_name: str, err_val: float) -> str:
    nicenames = {
        "amount": "amount was unusual",
        "amount_log": "amount (log) was unusual",
        "amount_sign": "transaction direction was unusual",
        "posting_lag_days": "posting lag was unusual",
        "same_amount_count_per_day": "same-amount count today was unusual",
        "txn_count_7d": "7-day txn volume was unusual",
        "txn_count_30d": "30-day txn volume was unusual",
        "mean_amount_30d": "30-day avg amount was unusual",
        "std_amount_30d": "30-day volatility was unusual",
        "zscore_amount_30d": "amount was far from 30-day pattern",
        "day_of_week": "day-of-week pattern was unusual",
        "has_matched_ref": "matched reference looked unusual",
        "cashbook_flag": "cashbook flag looked unusual",
    }
    if feat_name in nicenames:
        return f"{nicenames[feat_name]} (err={err_val:.4f})"

    # handle one-hot like BankTransactionCode_XXX
    if "_" in feat_name:
        prefix, rest = feat_name.split("_", 1)
        if prefix in ("BankAccountCode", "BankTransactionCode", "BusinessUnitCode", "CashBookFlag"):
            return f"{prefix}='{rest}' contributed most (err={err_val:.4f})"

    return f"{feat_name} contributed most (err={err_val:.4f})"



# --- per-feature deviation (which feature deviated most) ---

# squared error per feature
valid_sqerr = (X_valid - pred_valid) ** 2
train_sqerr = (X_train - pred_train) ** 2

# index of feature with max error per row
valid_max_idx = np.argmax(valid_sqerr, axis=1)
train_max_idx = np.argmax(train_sqerr, axis=1)

# map to names
valid_top_feat = [feat_names[i] for i in valid_max_idx]
train_top_feat = [feat_names[i] for i in train_max_idx]

# pick the actual error value for that feature
valid_top_feat_err = valid_sqerr[np.arange(len(valid_max_idx)), valid_max_idx]
train_top_feat_err = train_sqerr[np.arange(len(train_max_idx)), train_max_idx]

# also capture actual vs predicted (NOTE: scaled space)
valid_top_actual = X_valid[np.arange(len(valid_max_idx)), valid_max_idx]
valid_top_pred   = pred_valid[np.arange(len(valid_max_idx)), valid_max_idx]

train_top_actual = X_train[np.arange(len(train_max_idx)), train_max_idx]
train_top_pred   = pred_train[np.arange(len(train_max_idx)), train_max_idx]

# build human-ish reasons
valid_reasons = [
    make_reason_from_feature(f, e)
    for f, e in zip(valid_top_feat, valid_top_feat_err)
]
train_reasons = [
    make_reason_from_feature(f, e)
    for f, e in zip(train_top_feat, train_top_feat_err)
]

#===========

feats_train_out = feats_train.copy()
feats_train_out["recon_error"] = train_err
feats_train_out["top_feature"] = train_top_feat
feats_train_out["top_feature_error"] = train_top_feat_err
feats_train_out["top_feature_actual_scaled"] = train_top_actual
feats_train_out["top_feature_pred_scaled"] = train_top_pred
feats_train_out["contrib_reason"] = train_reasons
feats_train_out.to_csv(os.path.join(OUT_DIR, "train_engineered_with_scores.csv"), index=False)

# ===========

feats_valid_out = feats_valid.copy()
feats_valid_out["recon_error"] = valid_err
feats_valid_out["top_feature"] = valid_top_feat
feats_valid_out["top_feature_error"] = valid_top_feat_err
feats_valid_out["top_feature_actual_scaled"] = valid_top_actual
feats_valid_out["top_feature_pred_scaled"] = valid_top_pred
feats_valid_out["contrib_reason"] = valid_reasons
feats_valid_out.to_csv(os.path.join(OUT_DIR, "valid_engineered_with_scores.csv"), index=False)

# ==========





