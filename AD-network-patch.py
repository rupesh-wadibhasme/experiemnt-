# ----- inputs & embeddings -----
cat_inputs, cat_embeds = [], []
for i, (k, d) in enumerate(zip(cardinals, embed_dims)):
    inp = Input(shape=(1,), dtype="int32", name=f"cat{i}_in")

    # ⬇️  SPECIAL CASE for cat2, cat3, cat4  ⬇️
    if i in (2, 3, 4):
        d = max(2, d // 2)                         # shrink dimension
        emb = layers.Embedding(
            input_dim=k,
            output_dim=d,
            name=f"cat{i}_emb",
            embeddings_regularizer=keras.regularizers.l2(1e-4)  # L2 reg
        )(inp)
    # ⬆️  all other categoricals keep the original spec ⬆️
    else:
        emb = layers.Embedding(k, d, name=f"cat{i}_emb")(inp)

    cat_inputs.append(inp)
    cat_embeds.append(layers.Flatten()(emb))


'''
# 5-C.  Reconstruction & anomaly scores ─────────────────────────────
# Build inputs from the ENTIRE dataset (train + val) so shapes match
cat_all, num_all, _, _ = prepare_blocks(blocks, numeric_block_idx)
inputs_all = cat_all + [num_all]          # len == 7 → matches model.inputs

# Predict
recon = model.predict(inputs_all, batch_size=512, verbose=0)

# Compute errors
err_groups, row_score = compute_errors(cat_all, num_all, recon)



'''


def build_anomaly_pair(train_blocks,
                       train_idx,
                       row_score_train,
                       elbow_score,
                       loss_col="recon_loss"):
    """
    Returns
    -------
    [ df_anom, df_train_orig ]  (both pandas DataFrames)
        df_anom        – rows whose reconstruction error > elbow_score,
                          plus a new column `recon_loss`
        df_train_orig  – the full training-set DataFrame (no loss column)
    """
    # 1️⃣  stitch the training blocks back together
    df_train_orig = pd.concat(train_blocks, axis=1).reset_index(drop=True)

    # 2️⃣  find which training rows exceed the elbow
    anom_mask = row_score_train > elbow_score
    df_anom   = df_train_orig.loc[anom_mask].copy()
    df_anom[loss_col] = row_score_train[anom_mask]

    print(f"Anomalies in train split: {len(df_anom):,} "
          f"out of {len(df_train_orig):,} rows")

    return [df_anom, df_train_orig]


# row_score_train was produced via:
# recon_tr  = model.predict(inputs_tr, ...)
# err_grp_tr, row_score_train = compute_errors(cat_arrays_tr, num_array_tr, recon_tr)

# ─── 6.  Save elbow-anomalies for downstream steps ─────────────────────────

# 6-A.  Build the full DataFrame (all original columns side-by-side)
df_full = pd.concat(blocks, axis=1).reset_index(drop=True)

# 6-B.  Add the reconstruction-loss column you just computed
df_full["reconstruction_loss"] = row_score

# 6-C.  Filter rows above the elbow threshold
elbow_mask    = row_score > elbow_score          # ← elbow_score from earlier
df_anomalies  = df_full.loc[elbow_mask].copy()

print(f"Rows flagged by elbow: {len(df_anomalies):,}")

# 6-D.  Persist the result
df_anomalies.to_csv("elbow_anomaly_rows.csv", index=False)
print("Saved → elbow_anomaly_rows.csv")



anom_list = build_anomaly_pair(
                train_blocks=train_blocks,
                train_idx=train_idx,
                row_score_train=row_score_train,
                elbow_score=elbow_score,   # from the earlier elbow calc
            )

df_anom, df_train_orig = anom_list        # unpack if you like

# Optional: save just the anomalies
df_anom.to_csv("train_elbow_anomalies.csv", index=False)
