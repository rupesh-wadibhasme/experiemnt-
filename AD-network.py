"""
Auto-encoder with learnable embeddings for “list-of-DataFrames” input.
 - One model, eight inputs  (7 categorical one-hot blocks, 1 numeric block)
 - Per-block reconstruction, per-row anomaly score
 - Training/validation loss curves (overall + per block)

Author: ChatGPT • 2025-06-11
"""

# ────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple

keras   = tf.keras
layers  = keras.layers
Input   = layers.Input
Dense   = layers.Dense


# ────────────────────────────────────────────────────────────────────────────
# 1. Data-prep helpers
# ────────────────────────────────────────────────────────────────────────────
def onehot_df_to_index(df: pd.DataFrame) -> np.ndarray:
    """(N,k) one-hot → (N,1) int32 index array."""
    return df.values.argmax(axis=1).astype("int32")[:, None]

def prepare_blocks(
        blocks: List[pd.DataFrame],
        numeric_block_idx: int,
        embed_dim_rule=lambda k: max(2, int(np.ceil(np.sqrt(k))))
) -> Tuple[List[np.ndarray], np.ndarray, List[int], List[int]]:
    """
    Converts the list of DataFrames into:
        cat_arrays   – list of (N,1) int32 arrays  (categoricals)
        num_array    – (N,num_dim) float32 array   (numeric block)
        cardinals    – vocab sizes per categorical block
        embed_dims   – embedding size per categorical block
    """
    cat_arrays, cardinals, embed_dims = [], [], []
    for i, df in enumerate(blocks):
        if i == numeric_block_idx:
            num_array = df.values.astype("float32")
            continue
        arr = onehot_df_to_index(df)
        cat_arrays.append(arr)
        cardinals.append(df.shape[1])
        embed_dims.append(embed_dim_rule(df.shape[1]))
    return cat_arrays, num_array, cardinals, embed_dims


# ────────────────────────────────────────────────────────────────────────────
# 2. Model-builder
# ────────────────────────────────────────────────────────────────────────────
def build_embed_autoencoder(cardinals: List[int],
                            num_dim: int,
                            embed_dims: List[int],
                            hid: int = 64,
                            bottleneck: int = 32,
                            dropout: float = 0.15) -> keras.Model:
    """
    cardinals  – e.g. [3,4,3,30,18,18,18]
    num_dim    – e.g. 7
    embed_dims – e.g. [2,3,2,6,4,4,4]
    """
    # ----- inputs & embeddings -----
    cat_inputs, cat_embeds = [], []
    for i, (k, d) in enumerate(zip(cardinals, embed_dims)):
        inp = Input(shape=(1,), dtype="int32", name=f"cat{i}_in")
        emb = layers.Embedding(k, d, name=f"cat{i}_emb")(inp)
        cat_inputs.append(inp)
        cat_embeds.append(layers.Flatten()(emb))

    num_input = Input(shape=(num_dim,), name="num_in")

    # ----- concatenate -----
    x = layers.Concatenate()(cat_embeds + [num_input])
    x = layers.BatchNormalization()(x)

    # ----- encoder -----
    x = Dense(hid, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    code = Dense(bottleneck, activation="relu", name="bottleneck")(x)

    # ----- decoder -----
    y = Dense(hid, activation="relu")(code)

    cat_outputs = [
        Dense(k, activation="softmax", name=f"cat{i}_out")(y)
        for i, k in enumerate(cardinals)
    ]
    num_out = Dense(num_dim, activation="linear", name="num_out")(y)

    model = keras.Model(inputs=cat_inputs + [num_input],
                        outputs=cat_outputs + [num_out])

    losses = (
        [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
        + [keras.losses.MeanSquaredError()]
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss=losses)
    return model


# ────────────────────────────────────────────────────────────────────────────
# 3. Error computation
# ────────────────────────────────────────────────────────────────────────────
def compute_errors(cat_true, num_true, recon):
    """
    Returns:
        err_groups – list of (N, k_i) arrays  (k_i = 1 for cats, num_dim for nums)
        row_score  – (N,) overall anomaly score
    """
    err_groups = []

    # ❶  categorical blocks: 0 / 1 mismatch, keep as (N,1)
    for true, pred in zip(cat_true, recon[:-1]):
        mismatch = (pred.argmax(-1) != true.squeeze()).astype("float32")[:, None]
        err_groups.append(mismatch)                     # shape (N,1)

    # ❷  numeric block: squared error → z-score, already (N,num_dim)
    num_err = (recon[-1] - num_true) ** 2
    num_err = (num_err - num_err.mean(0)) / (num_err.std(0) + 1e-9)
    err_groups.append(num_err)                         # shape (N,7)

    # ❸  concatenate along feature axis, then sum to one scalar per row
    row_score = np.concatenate(err_groups, axis=1).sum(1)
    return err_groups, row_score

# ────────────────────────────────────────────────────────────────────────────
# 4. Training-history plots
# ────────────────────────────────────────────────────────────────────────────
def plot_history(hist: dict, block_names: List[str]):
    """Plot total loss + each block’s loss."""
    # total
    plt.figure(figsize=(6, 4))
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Total loss"); plt.legend(); plt.show()

    # per block
    for name in block_names:
        tr = f"{name}_out_loss"
        va = f"val_{name}_out_loss"
        if tr in hist:
            plt.figure(figsize=(5, 3))
            plt.plot(hist[tr], label="train")
            plt.plot(hist[va], label="val")
            plt.title(name); plt.legend(); plt.show()

#------------------
import numpy as np
import matplotlib.pyplot as plt

def plot_sorted_errors(row_score, top_fraction=0.02, show_gaps=True):
    """
    row_score    – 1-D array of reconstruction errors (one per row)
    top_fraction – 0.02 ⇒ top-2 %; 0.01 ⇒ top-1 %; etc.
    show_gaps    – set False if you only want the main curve
    """
    # 1. sort the errors
    sorted_scores = np.sort(row_score)
    N             = len(sorted_scores)

    # 2. percentile cutoff
    cutoff_idx   = int(np.ceil((1 - top_fraction) * N)) - 1
    cutoff_score = sorted_scores[cutoff_idx]

    # 3. optional “largest gap” (elbow) finder
    deltas      = np.diff(sorted_scores)
    elbow_idx   = np.argmax(deltas)
    elbow_score = sorted_scores[elbow_idx]

    # 4. PLOT — sorted errors
    plt.figure(figsize=(7, 4))
    plt.plot(sorted_scores, lw=2, label="sorted error")
    plt.axvline(cutoff_idx,  ls="--", color="tab:orange",
                label=f"{100*(1-top_fraction):.0f}th-percentile")
    plt.axhline(cutoff_score, ls="--", color="tab:orange")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index (after sorting)")
    plt.ylabel("Reconstruction error")
    plt.legend(); plt.tight_layout(); plt.show()

    # 5. PLOT — consecutive gaps (optional)
    if show_gaps:
        plt.figure(figsize=(7, 3))
        plt.plot(deltas, lw=2, label="error difference i→i+1")
        plt.axvline(elbow_idx, ls="--", color="tab:red",
                    label="largest jump (elbow)")
        plt.title("Gap between consecutive sorted errors")
        plt.xlabel("Index (between i and i+1)")
        plt.ylabel("Δ error")
        plt.legend(); plt.tight_layout(); plt.show()

    return cutoff_score, cutoff_idx, elbow_score, elbow_idx


# ────────────────────────────────────────────────────────────────────────────
# 5. Main pipeline (wrap in `if __name__ == "__main__":` if desired)
# ────────────────────────────────────────────────────────────────────────────

# 5-A.  Your data ------------------------------------------------------------
# blocks = [df0, df1, df2, df3, df4, df5, df6, df7]  # supply these yourself
numeric_block_idx = 7                                # adjust if needed

# 5-A.  Your data ------------------------------------------------------------
# blocks = [...]  # your list of 8 DataFrames  (cat0 … cat6, numerics)
numeric_block_idx = 7                   # keep if block 7 is the numeric one

# ▼▼▼  NEW: stratify on cat2 & cat3  ▼▼▼
from sklearn.model_selection import train_test_split
import numpy as np

idx        = np.arange(len(blocks[0]))
cat2_lbl   = blocks[2].values.argmax(1)
cat3_lbl   = blocks[3].values.argmax(1)
combo_lbl  = cat2_lbl * 1000 + cat3_lbl

# 1️⃣  count occurrences per combo
combo_counts = np.bincount(combo_lbl)
rare_mask    = combo_counts[combo_lbl] < 2     # rows whose combo appears < 2×

# 2️⃣  always put rare rows into train set
common_idx   = idx[~rare_mask]                 # safe to stratify
rare_idx     = idx[rare_mask]                  # will stay in train

# 3️⃣  stratified split on the common part only
tr_c, va_c = train_test_split(
    common_idx,
    test_size=0.25,
    random_state=42,
    stratify=combo_lbl[common_idx]
)

# 4️⃣  final indices
train_idx = np.concatenate([tr_c, rare_idx])   # train = common-train + all rare
val_idx   = va_c                               # validation = common-val only

print(f"Train rows: {len(train_idx):,}  |  Val rows: {len(val_idx):,}"
      f"  |  Rare rows forced to train: {len(rare_idx)}")


train_blocks = [df.iloc[train_idx].reset_index(drop=True) for df in blocks]
val_blocks   = [df.iloc[val_idx]  .reset_index(drop=True) for df in blocks]
# ▲▲▲  END new code  ▲▲▲

# prepare ⇢ TRAIN
cat_arrays_tr, num_array_tr, cardinals, embed_dims = prepare_blocks(
        train_blocks, numeric_block_idx)
inputs_tr  = cat_arrays_tr + [num_array_tr]
targets_tr = inputs_tr

# prepare ⇢ VALIDATION
cat_arrays_va, num_array_va, _, _ = prepare_blocks(
        val_blocks, numeric_block_idx)
inputs_va  = cat_arrays_va + [num_array_va]
targets_va = inputs_va

# 5-B.  Build & train --------------------------------------------------------
model = build_embed_autoencoder(cardinals,
                                num_dim=num_array_tr.shape[1],
                                embed_dims=embed_dims,
                                hid=64, bottleneck=32, dropout=0.15)

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=20,
                                           restore_best_weights=True)

history = model.fit(inputs_tr, targets_tr,                 # ← train data
                    validation_data=(inputs_va, targets_va),  # ← val data
                    epochs=200,
                    batch_size=128,
                    callbacks=[early_stop],
                    verbose=2)


# 5-C.  Reconstruction & anomaly scores -------------------------------------
recon = model.predict(inputs, batch_size=512, verbose=0)
err_groups, row_score = compute_errors(cat_arrays, num_array, recon)

# example: flag top 2 % as anomalies
threshold = np.quantile(row_score, 0.98)
anomaly_mask = row_score > threshold
print(f"Anomalies detected: {anomaly_mask.sum()} / {len(row_score)}")

# `row_score` is produced right after compute_errors(...)
cutoff_score, cutoff_idx, elbow_score, elbow_idx = plot_sorted_errors(
        row_score,
        top_fraction=0.01      # ← 0.01 for top-1 %; change to 0.02 for top-2 %
)

# flag anomalies
anomaly_mask = row_score > cutoff_score
print(f"Flagged {anomaly_mask.sum()} / {len(row_score)} rows "
      f"(top {100*0.01:.0f} %).")


# 5-D.  Plot learning curves -------------------------------------------------
block_names = [f"cat{i}" for i in range(len(cardinals))] + ["num"]
plot_history(history.history, block_names)
