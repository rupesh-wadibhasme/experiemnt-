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




# ----------------------------------------------------------------------
# Build reconstructed DataFrames and *then* inverse-transform
#   specified numeric columns with a single MinMaxScaler.
# ----------------------------------------------------------------------
def build_reconstructed_dataframes(blocks,
                                   recon,
                                   numeric_block_idx,
                                   numeric_cols: list,
                                   num_scaler):
    """
    Parameters
    ----------
    blocks             : list[pd.DataFrame]
    recon              : list[np.ndarray]  (model.predict output)
    numeric_block_idx  : int   – position of the numeric DataFrame in blocks
    numeric_cols       : list[str]  – column names to inverse-transform
                                      (order must match scaler fitting order)
    num_scaler         : fitted MinMaxScaler (trained on numeric_cols)

    Returns
    -------
    df_recon_1hot : pandas DataFrame  – categorical 0/1, numerics inverse-scaled
    df_recon_prob : pandas DataFrame  – categorical probabilities, numerics inverse-scaled
    """
    recon_1hot_blocks, recon_prob_blocks = [], []
    r_iter = iter(recon)                     # iterator over cat logits

    for i, df in enumerate(blocks):
        if i == numeric_block_idx:
            # numeric block → add *scaled* for now
            num_pred_scaled = pd.DataFrame(recon[-1], columns=df.columns)
            recon_1hot_blocks.append(num_pred_scaled)
            recon_prob_blocks.append(num_pred_scaled.copy())
        else:
            logits   = next(r_iter)
            preds    = logits.argmax(-1)
            one_hot  = np.eye(df.shape[1])[preds]
            cat_1hot = pd.DataFrame(one_hot, columns=df.columns).astype(int)
            cat_prob = pd.DataFrame(logits,  columns=df.columns)

            recon_1hot_blocks.append(cat_1hot)
            recon_prob_blocks.append(cat_prob)

    # --- concatenate blocks into full DataFrames (still scaled numerics) ----
    df_recon_1hot = pd.concat(recon_1hot_blocks, axis=1).reset_index(drop=True)
    df_recon_prob = pd.concat(recon_prob_blocks, axis=1).reset_index(drop=True)
    df_original= pd.concat(blocks, axis=1).reset_index(drop=True)

    # --- inverse-transform *only* the requested numeric columns -------------
    for df_tmp in (df_recon_1hot, df_recon_prob,df_original):
        scaled_vals = df_tmp[numeric_cols].values          # (N, num_dim)
        raw_vals    = num_scaler.inverse_transform(scaled_vals)
        df_tmp[numeric_cols] = raw_vals                    # overwrite in place

    return df_recon_1hot, df_recon_prob,df_original
