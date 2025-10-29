def time_based_split(feats_df: pd.DataFrame, ts_col="ts", cutoff=None, valid_frac=0.15):
    """
    Return boolean masks aligned to the ORIGINAL feats_df order.
    - If cutoff is given, we do a simple comparison in-place (no sorting).
    - If valid_frac is used, we sort only to pick which rows belong to train,
      then we map those indices back into the original order to build masks.
    """
    ts = pd.to_datetime(feats_df[ts_col])

    if cutoff:
        cut = pd.to_datetime(cutoff)
        train_mask = (ts < cut).values
        valid_mask = ~train_mask
        return train_mask, valid_mask

    # fraction-based split: decide by time order, but build masks in original order
    order = np.argsort(ts.values)               # positions that would sort by time
    n = len(feats_df)
    n_train = int(np.floor((1.0 - valid_frac) * n))
    train_idx_sorted = order[:n_train]          # earliest rows by time
    mask = np.zeros(n, dtype=bool)
    mask[train_idx_sorted] = True               # mark those positions in ORIGINAL order
    train_mask = mask
    valid_mask = ~mask
    return train_mask, valid_mask
