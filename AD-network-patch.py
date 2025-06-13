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


# ─── 1.  elbow score ────────────────────────────────────────────────
sorted_scores = np.sort(row_score)
deltas        = np.diff(sorted_scores)
elbow_idx     = np.argmax(deltas)
elbow_score   = sorted_scores[elbow_idx]

print(f"Elbow at index {elbow_idx:,}  →  score {elbow_score:.6f}")

# ─── 2.  mask of rows above elbow ───────────────────────────────────
elbow_mask = row_score > elbow_score
print(f"Rows flagged as anomalies: {elbow_mask.sum():,} "
      f"({100*elbow_mask.mean():.2f} % of dataset)")

# ─── 3.  rebuild full DataFrame ─────────────────────────────────────
df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)     # original cols
df_anom = df_all.loc[elbow_mask].copy()                        # subset

# attach the loss column only to the anomaly DF
df_anom["reconstruction_loss"] = row_score[elbow_mask]

# ─── 4.  save and/or return ─────────────────────────────────────────
df_anom.to_csv("elbow_anomalies.csv", index=False)
print("Saved → elbow_anomalies.csv")

# pack both frames in a list for downstream use
anom_bundle = [df_anom, df_all]    # anom_bundle[0] = anomalies, [1] = original

#--------------------------------------
def scale(df, scaler=None):
  if scaler is None:
    scaler = MinMaxScaler()
    scaler.fit(df)
  scaled_data = scaler.transform(df)
  scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(-1)
  return scaled_df, scaler




# ------------------------------------------------------------------
# Re-build two reconstructed DataFrames
#   • df_recon_1hot  – cats as one-hot 0/1, numeric in ORIGINAL units
#   • df_recon_prob  – cats as soft-max probabilities, numeric in ORIGINAL units
# ------------------------------------------------------------------
def build_reconstructed_dataframes(blocks,
                                   recon,
                                   numeric_block_idx=6,
                                   num_scaler=None):
    """
    Parameters
    ----------
    blocks             : list[pd.DataFrame]
    recon              : list[np.ndarray]  (model.predict output)
    numeric_block_idx  : int               (position of numeric DF in blocks)
    num_scaler         : fitted MinMaxScaler (or compatible) ―
                         if provided, numeric block is inverse-transformed
                         back to real units; otherwise stays scaled.

    Returns
    -------
    df_recon_1hot, df_recon_prob   (pandas DataFrames)
    """

    recon_1hot_blocks, recon_prob_blocks = [], []
    r_iter = iter(recon)  # iterate over categorical heads in order

    for i, df in enumerate(blocks):
        if i == numeric_block_idx:
            # ---- numeric block ------------------------------------
            num_pred = recon[-1]                          # ndarray (N, num_dim)
            if num_scaler is not None:
                num_pred = num_scaler.inverse_transform(num_pred)
            num_pred_df = pd.DataFrame(num_pred, columns=df.columns)

            recon_1hot_blocks.append(num_pred_df)
            recon_prob_blocks.append(num_pred_df.copy())

        else:
            # ---- categorical block -------------------------------
            logits   = next(r_iter)                       # shape (N, k)
            # one-hot (0/1) version
            preds    = logits.argmax(-1)
            one_hot  = np.eye(df.shape[1])[preds]
            cat_1hot = pd.DataFrame(one_hot, columns=df.columns).astype(int)
            # probability version
            cat_prob = pd.DataFrame(logits, columns=df.columns)

            recon_1hot_blocks.append(cat_1hot)
            recon_prob_blocks.append(cat_prob)

    df_recon_1hot = pd.concat(recon_1hot_blocks, axis=1).reset_index(drop=True)
    df_recon_prob = pd.concat(recon_prob_blocks, axis=1).reset_index(drop=True)
    return df_recon_1hot, df_recon_prob
