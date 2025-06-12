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
