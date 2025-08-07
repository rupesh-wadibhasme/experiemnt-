
import warnings
warnings.filterwarnings('ignore')

numeric_columns =['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints', 'Is_weekend_date','TDays','FaceValue']
categorical_columns =['Instrument', 'BUnit', 'Cpty',
                      'PrimaryCurr', 'BuyCurr', 'SellCurr'
                      ]

# # Helper functions ........

# ────────────────────────────────────────────────────────────────────────────
# 4. Training-history plots
# ────────────────────────────────────────────────────────────────────────────
from typing import List, Tuple
import pandas as pd 
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



def plot_error_with_quantile_lines(row_score,
                                   quantiles=(0.01, 0.05, 0.10),
                                   line_kw=None,
                                   figsize=(7, 4)):
    """
    Draw the sorted-error curve and add vertical dotted lines that mark
    the top-q fraction of highest-error rows.  Prints how many rows fall
    in each tail.

    """
    if line_kw is None:
        line_kw = dict(ls=":", lw=1.5, color="tab:red")

    # sort ascending so largest errors are on the right
    sorted_scores = np.sort(row_score)
    N             = len(sorted_scores)

    plt.figure(figsize=figsize)
    plt.plot(sorted_scores, lw=2)
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index (after sorting ↑)")
    plt.ylabel("Error")

    # print header
    print("Tail summary:")
    print(f"{'Quantile':>9} | {'Rows':>6}")

    # add one dotted line per quantile
    for q in sorted(quantiles):
        idx   = int(np.ceil((1 - q) * N)) - 1          # right-hand start index
        nrows = N - idx                                # how many rows in the tail

        # vertical line + text label on plot
        plt.axvline(idx, **line_kw)
        plt.text(idx, sorted_scores[0],
                 f" {int(q*100)}% ({nrows})", rotation=90,
                 va="bottom", ha="right", fontsize=8)

        # console print-out
        print(f"{q:>7.0%} | {nrows:6}")

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------
# Build reconstructed DataFrames and *then* inverse-transform
#   specified numeric columns with a single MinMaxScaler.
# ----------------------------------------------------------------------
def build_reconstructed_dataframes(blocks,
                                   recon,
                                   numeric_block_idx,
                                   numeric_cols: list
                                   ,row_score,
                                   save=True):
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

    # --- concatenate blocks into full DataFrames (still scaled numerics) ---

    df_recon_1hot = pd.concat(recon_1hot_blocks, axis=1).reset_index(drop=True)
    df_recon_prob = pd.concat(recon_prob_blocks, axis=1).reset_index(drop=True)
    df_original= pd.concat(blocks, axis=1).reset_index(drop=True)

    if save==True:
        df_recon_1hot['recon_error']=row_score
        df_recon_1hot.to_csv('unscaled_prediction_onehot.csv')
        df_original.to_csv('unscaled_original.csv')
  
    if save==True:
        df_recon_1hot.to_csv('scaled_prediction_onehot.csv')
        df_original.to_csv('scaled_original.csv')
        
    return df_recon_1hot, df_recon_prob,df_original

# # 1. Configuration

# # *** Encoder model ***

# # Helper functions for Tuning AutoEncoder

# ────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple
import tensorflow.keras.backend as K
from keras.src.layers import Input, Dense,Concatenate,BatchNormalization,Dropout,GaussianNoise
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.losses import CategoricalCrossentropy
from keras.src.callbacks import CSVLogger,EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    cardinals   e.g. [3,4,3,30,18,18,18]
    num_dim     e.g. 7
    embed_dims  e.g. [2,3,2,6,4,4,4]
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

    def perc_mse(y_true, y_pred):
        pct = abs(y_pred - y_true) / (K.maximum(K.abs(y_true), 1e-9))
        return K.mean(K.square(pct), axis=-1)
    

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
        err_groups  list of (N, k_i) arrays  (k_i = 1 for cats, num_dim for nums)
        row_score   (N,) overall anomaly score
    """
    err_groups = []

    # ❶  categorical blocks: 0 / 1 mismatch, keep as (N,1)
    for true, pred in zip(cat_true, recon[:-1]):
        mismatch = (pred.argmax(-1) != true.squeeze()).astype("float32")[:, None]
        err_groups.append(mismatch)                     # shape (N,1)

    # ❷  numeric block: squared error → z-score, already (N,num_dim)
    num_err = (recon[-1] - num_true) ** 2
    num_err = abs(num_err - num_err.mean(0)) / (num_err.std(0) + 1e-9)
    err_groups.append(num_err)                         # shape (N,7)

    # ❸  concatenate along feature axis, then sum to one scalar per row
    row_score = np.concatenate(err_groups, axis=1).sum(1)
    return err_groups, row_score


# LOAD DATASET 

import pickle

with open(r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\processed_training_data\IXOM_training_data.pkl', 'rb') as f:
    input_data = pickle.load(f)



# # Train Autoencoder

blocks=input_data
numeric_block_idx = 6
# ▼▼▼  NEW: stratify on cat2 & cat3  ▼▼▼
# ▼▼▼  NEW: stratify on cat2 & cat3  ▼▼▼
from sklearn.model_selection import train_test_split
import numpy as np

idx        = np.arange(len(blocks[0]))
cat2_lbl   = blocks[1].values.argmax(1)
cat3_lbl   = blocks[2].values.argmax(1)
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
    test_size=0.30,
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
                                           patience=5,
                                           restore_best_weights=True)

history = model.fit(inputs_tr, targets_tr,                 # ← train data
                    validation_data=(inputs_va, targets_va),  # ← val data
                    epochs=100,
                    batch_size=64,
                    callbacks=[early_stop],
                    verbose=2)

# # Save keras model


model.save(r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_keras_models\IXOM_latest_model_2.keras") 

#original_train
# 5-C.  Reconstruction & anomaly scores -------------------------------------
cat_all, num_all, _, _ = prepare_blocks(blocks, numeric_block_idx)
inputs_all = cat_all + [num_all]          # len == 7 → matches model.inputs


recon = model.predict(inputs_all, batch_size=512, verbose=0)

err_groups, row_score = compute_errors(cat_all, num_all, recon)

plot_error_with_quantile_lines(row_score,quantiles=(0.01,0.03,0.05,0.10))

df_recon_1hot, df_recon_prob, df_original =build_reconstructed_dataframes(blocks,recon,numeric_block_idx=6,numeric_cols=numeric_columns[:-1],row_score=row_score)


# 5-D.  Plot learning curves -------------------------------------------------
block_names = [f"cat{i}" for i in range(len(cardinals))] + ["num"]
plot_history(history.history, block_names)

# # Validate anomaly count 

counter_1 = 0
counter_2 = 0
counter_z = 0
non_anomaly = 0
thresh_one = 0.80
thresh_two = 0.90
fts = df_original
print(fts.shape)
df_deviation = fts-df_recon_prob
for idx, row in df_deviation.iterrows():
    filtered_columns = row[row > thresh_one].index.tolist()
    filtered_columns = [x for x in filtered_columns if not (x.startswith('BUnit') or x.startswith('Cpty'))]
        
    filtered_columns_2 = row[row > thresh_two].index.tolist()
    filtered_columns_2 = [x for x in filtered_columns_2 if not (x.startswith('BUnit') or x.startswith('Cpty'))]
    if len(filtered_columns)>0:
        counter_1+=1
    elif len(filtered_columns_2)>2:
        counter_2+=1
        # elif get_zscore(row['FaceValue'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['mean'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['sd'])>3 or get_zscore(row['FaceValue'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['mean'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['sd'])<-3:
    #     counter_z+=1
    else:
        non_anomaly+=1
print('Atleast one greater than 0.995 threshold:', counter_1) ## 7592
print('Atleast two greater than 0.95 threshold:', counter_2) ## 3645
print('non_anomalous:', non_anomaly) # 10441


