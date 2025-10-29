def time_based_split_indices(feats_df: pd.DataFrame, ts_col="ts", cutoff=None, valid_frac=0.15):
    """
    Return integer index arrays (train_idx, valid_idx) w.r.t. the ORIGINAL feats_df order.
    """
    ts = pd.to_datetime(feats_df[ts_col]).to_numpy()

    if cutoff:
        cut = np.datetime64(pd.to_datetime(cutoff))
        train_idx = np.flatnonzero(ts < cut)
        valid_idx = np.flatnonzero(ts >= cut)
        return train_idx, valid_idx

    # fraction-based split decided by time order, then mapped back to original positions
    order = np.argsort(ts)               # positions that would sort by time
    n = len(ts)
    n_train = int(np.floor((1.0 - valid_frac) * n))
    train_idx = order[:n_train]
    valid_idx = order[n_train:]
    return train_idx, valid_idx



# --- split to indices ---
train_idx, valid_idx = time_based_split_indices(feats_all, ts_col="ts",
                                                cutoff=CUTOFF, valid_frac=VALID_FRAC)

# --- slice by indices (keeps X and feats aligned) ---
X_train  = X_all[train_idx]
X_valid  = X_all[valid_idx]
feats_train = feats_all.iloc[train_idx].reset_index(drop=True)
feats_valid = feats_all.iloc[valid_idx].reset_index(drop=True)

# --- optional burn-in filter as indices (NO boolean mask length issues) ---
if BURNIN_TXN30D_MIN > 0 and "txn_count_30d" in feats_train.columns:
    keep_idx = np.flatnonzero(feats_train["txn_count_30d"].to_numpy() >= BURNIN_TXN30D_MIN)
    X_train      = X_train[keep_idx]
    feats_train  = feats_train.iloc[keep_idx].reset_index(drop=True)

# --- sanity checks (fail fast if anything drifts) ---
assert len(X_all) == len(feats_all), "X_all and feats_all length mismatch"
assert len(X_train) == len(feats_train), "Post-split TRAIN length mismatch"
assert len(X_valid) == len(feats_valid), "Post-split VALID length mismatch"
