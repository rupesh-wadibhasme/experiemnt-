def flag_extra_volume_txns_with_baseline(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    account_col: str = "BankAccountCode",
    bu_col: str = "BusinessUnitCode",
    code_col: str = "BankTransactionCode",
    posting_date_col: str = "PostingDateKey",
    txn_id_col: str = "BankTransactionId",   # which column holds transaction IDs
    min_history_days: int = 5,
    zscore_threshold: float = 3.0,
    sort_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Use PRECOMPUTED baseline stats to flag only the "extra" transactions
    on high-volume days.

    Adds to df:
      - group_count_today
      - mean_daily_count
      - std_daily_count
      - active_days
      - is_volume_spike_txn   (0/1 per row)
      - volume_spike_reason   (text for flagged rows)
      - group_txn_ids         (for flagged rows: ALL BankTransactionId values
                               for that (acct, BU, code, date) group)
    """
    if posting_date_col not in df.columns:
        raise KeyError(f"Column '{posting_date_col}' not found in dataframe")

    if txn_id_col not in df.columns:
        raise KeyError(f"Column '{txn_id_col}' not found in dataframe (needed for group_txn_ids).")

    out = df.copy()

    # derive posting_date
    posting_ts = _to_datetime(out[posting_date_col])
    out["_posting_date"] = posting_ts.dt.date

    grp_keys = [account_col, bu_col, code_col, "_posting_date"]

    # daily counts on THIS dataset
    daily = (
        out.groupby(grp_keys, dropna=False)
           .size()
           .reset_index(name="group_count_today")
    )

    # join with baseline
    daily = daily.merge(
        baseline_df,
        on=[account_col, bu_col, code_col],
        how="left",
        validate="m:1",   # each group-day maps to one baseline row
    )

    # fill missing baseline (no history)
    daily["mean_daily_count"] = daily["mean_daily_count"].fillna(0.0)
    daily["std_daily_count"]  = daily["std_daily_count"].fillna(0.0)
    daily["active_days"]      = daily["active_days"].fillna(0).astype(int)

    # compute thresholds & extras
    mean_ = daily["mean_daily_count"].values
    std_  = daily["std_daily_count"].values
    cnt   = daily["group_count_today"].values
    act_days = daily["active_days"].values

    has_hist = act_days >= min_history_days
    std_pos  = std_ > 0

    # numeric threshold per group-day
    thresh = np.where(
        std_pos,
        mean_ + zscore_threshold * std_,
        mean_,
    )
    # integer threshold (at least 1)
    thresh_int = np.maximum(1, np.round(thresh).astype(int))

    extra = np.where(
        has_hist,
        np.maximum(0, cnt - thresh_int),
        0,
    )

    daily["threshold_int"] = thresh_int
    daily["extra_txn"] = extra

    # prepare output columns on row-level df
    out["group_count_today"]   = 0
    out["mean_daily_count"]    = 0.0
    out["std_daily_count"]     = 0.0
    out["active_days"]         = 0
    out["is_volume_spike_txn"] = 0
    out["volume_spike_reason"] = ""
    out["group_txn_ids"]       = ""   # ALL txn IDs for the group-day (for flagged rows)

    # per (acct, BU, code, date) with extra > 0
    for _, row in daily.iterrows():
        e = int(row["extra_txn"])
        if e <= 0:
            continue

        acct = row[account_col]
        bu   = row[bu_col]
        code = row[code_col]
        d    = row["_posting_date"]

        # mask for this group-day
        mask = (
            (out[account_col] == acct) &
            (out[bu_col] == bu) &
            (out[code_col] == code) &
            (out["_posting_date"] == d)
        )
        idx = out.index[mask]
        if len(idx) == 0:
            continue

        # attach stats to all rows for this group-day
        out.loc[mask, "group_count_today"] = int(row["group_count_today"])
        out.loc[mask, "mean_daily_count"]  = float(row["mean_daily_count"])
        out.loc[mask, "std_daily_count"]   = float(row["std_daily_count"])
        out.loc[mask, "active_days"]       = int(row["active_days"])

        # all txn IDs for this group-day (for context)
        group_ids = out.loc[idx, txn_id_col].astype(str).tolist()
        group_ids_str = ",".join(group_ids)

        # decide which specific rows are "extra"
        if sort_col is not None and sort_col in out.columns:
            idx_sorted = out.loc[idx].sort_values(sort_col).index
        else:
            # fall back to original order
            idx_sorted = idx.sort_values()

        # take last e rows as extra
        flag_idx = idx_sorted[-e:]

        # human-readable reason (volume part; non-biz will be merged later)
        base_reason = (
            f"This transaction is one of {e} extra txns above the typical "
            f"{int(row['threshold_int'])} per day for "
            f"(Account='{acct}', BU='{bu}', TxnCode='{code}'). "
            f"Today had {int(row['group_count_today'])} txns; "
            f"baseline≈{row['mean_daily_count']:.2f}±{row['std_daily_count']:.2f}."
        )

        if group_ids_str:
            base_reason += f" Other transactions to review for this group/day: [{group_ids_str}]."

        out.loc[flag_idx, "is_volume_spike_txn"] = 1
        out.loc[flag_idx, "volume_spike_reason"] = base_reason
        out.loc[flag_idx, "group_txn_ids"]       = group_ids_str

    # clean up helper
    out.drop(columns=["_posting_date"], inplace=True)
    return out
def combine_reasons_columns(
    df: pd.DataFrame,
    nonbiz_col: str = "nonbiz_reason",
    vol_col: str = "volume_spike_reason",
    out_col: str = "anomaly_reason",
) -> pd.DataFrame:
    """
    Merge non-business-day and volume-spike reasons into a single, numbered text column.

    If both exist:
        "1) <nonbiz_reason> 2) <volume_spike_reason>"
    If only one exists:
        "1) <that_reason>"
    If none:
        ""  (blank)
    """
    nb = df.get(nonbiz_col, pd.Series("", index=df.index)).fillna("").astype(str)
    vol = df.get(vol_col, pd.Series("", index=df.index)).fillna("").astype(str)

    combined = []
    for nb_text, vol_text in zip(nb, vol):
        nb_text = nb_text.strip()
        vol_text = vol_text.strip()
        parts = []
        if nb_text:
            parts.append(f"1) {nb_text}")
        if vol_text:
            idx = len(parts) + 1
            parts.append(f"{idx}) {vol_text}")
        combined.append(" ".join(parts))

    df[out_col] = combined
    return df
