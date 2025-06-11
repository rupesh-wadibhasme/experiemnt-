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
