# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # 1. Configuration

# %%
import pandas as pd
categorical_columns =['BUnit', 'Cpty','Instrument',
                      'PrimaryCurr', 'BuyCurr', 'SellCurr','ActivationDate','MaturityDate'
                      ]

numeric_columns =['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints']

groupby_columns_names = ['BUnit', 'Cpty','PrimaryCurr']
group_trantype = ['Instrument']

# %% [markdown]
# # 2. Data Sanity check (Drop Duplicates)

# %%

def data_sanity_check(data):
    # Drop the rows where BuyAmount and SellAmount both Zero
    try:
        data = data[(data.BuyAmount != 0.0) & (data.SellAmount != 0.0)]
        # Drop the columns that are discussed and decided based on functional team decision and correlation
        #data.drop(columns=cols_to_drop,axis=1,inplace=True)
        # Drop the duplicates
        data = data.drop_duplicates(keep='last')
        print(f'Size of data after dropping duplicate rows and cols: {len(data.index)}')
    except Exception as e:
        print(e)
    return data


# %% [markdown]
# # 3.Feature Engineering

# %%
import numpy as np
import traceback,sys

def face_value(df):
    if df.PrimaryCurr == df.BuyCurr:
        df["FaceValue"]=np.abs(df.BuyAmount)
    elif df.PrimaryCurr == df.SellCurr:
        df["FaceValue"]=np.abs(df.SellAmount)
    return df


def feature_engg(data,numeric_columns,categorical_columns):
    try:
        data['Is_weekend_date'] = data.ActivationDate.apply(lambda x: x.date().isoweekday())
        #Convert weekdays to '0' and weekend to '1'
        data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x<6 else 1)
        data['TDays'] = (data.MaturityDate - data.ActivationDate).dt.days

        data["FaceValue"] = np.nan
        data = data.apply(face_value, axis=1)


        #Convert BUnit & TranCode into Categorical.
        categories_in_numerics = ['BUnit']
        for col in categories_in_numerics:
            data[col] = data[col].astype('str')

        numeric_columns.append('Is_weekend_date')
        numeric_columns.append('TDays')
        numeric_columns.append('FaceValue')
        categorical_columns.remove('ActivationDate')
        categorical_columns.remove('MaturityDate')


    except Exception as e:
        print (e)

    return data,numeric_columns,categorical_columns

# %% [markdown]
# # 4. Grooping Scaling and Transformation

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
def minmax_scale_group(group, column):
  scaler = MinMaxScaler()
  group[[column]] = scaler.fit_transform(group[[column]])
  mean = float(group[column].mean())
  std_dev = float(group[column].std())
  return group, scaler, mean, std_dev

def clip_IQR(group, column):
  if len(group[column]) < 10:
    lower = group[column].min()
    upper = group[column].max()
    percent = 0
    counts = 0
    lower_counts = 0
    upper_counts = 0
  else:    
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3+1.5*IQR
    lower = Q1-1.5*IQR
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper, inclusive='both')].shape[0]
    percent = (counts*100)/group.shape[0]
    group[column] = group[column].clip(lower, upper)
  return group, lower, upper, (counts, lower_counts, upper_counts)

def clip_zscore(group, column, z_score=3):
  if len(group[column]) < 10:
    lower = group[column].min()
    upper = group[column].max()
    percent = 0
    counts = 0
    lower_counts = 0
    upper_counts = 0
  else:  
    mean = group[column].mean()
    std = group[column].std()
    value = mean + (z_score * std)
    lower = -value
    upper = value
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper, inclusive='both')].shape[0]
    percent = (counts*100)/group.shape[0]
    group[column] = group[column].clip(lower, upper)
  return group, lower, upper, (counts, lower_counts, upper_counts)

def clip_percentile(group, column, upper_percentile, lower_percentile=0.01):
  if len(group[column]) < 10:
    lower = group[column].min()
    upper = group[column].max()
    percent = 0
    counts = 0
    lower_counts = 0
    upper_counts = 0
  else:    
    upper = group[column].quantile(upper_percentile)
    lower = group[column].quantile(lower_percentile)
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper, inclusive='both')].shape[0]
    percent = (counts*100)/group.shape[0]
    group[column] = group[column].clip(lower =lower, upper = upper)
  return group, lower, upper, (counts, lower_counts, upper_counts)

def clip_mod_zscore(group, column, mod_zscore=3.5):
  if len(group[column])<10:
    lower = group[column].min()
    upper = group[column].max()
    counts = 0
    percent = 0
    lower_counts = 0
    upper_counts = 0
  else:
    median = group[column].median()
    mad = np.median(np.abs(group[column] - median))  
    lower = median - mod_zscore * mad / 0.6745
    upper = median + mod_zscore * mad / 0.6745
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper, inclusive='both')].shape[0]
    percent = (counts*100)/group.shape[0]
    group[column] = group[column].clip(lower=lower, upper=upper)
  return group, lower, upper, (counts, lower_counts, upper_counts)
 

def group_points(df, groupby_columns, column, iqr=False, zscore=False, percentile=False, mod_zscore=False):
  grouped_scalers = {}
  grouped_scaled_dfs = []
  grouped = df.groupby(groupby_columns, sort=False)
  iqr_bds = {}
  zscore_bds = {}
  percentile_lt = {}
  mod_zscore_lt = {}
  total_counts = 0
  lower_counts = 0
  upper_counts = 0
  for name, group in grouped:
      if iqr:
        group, lower, upper, counts = clip_IQR(group, column)
        iqr_bds[name] = {'lb':lower, 'ub':upper}
        # print(name, percent)
      if zscore:
        group, lower, upper, counts = clip_zscore(group, column)
        zscore_bds[name] = {'lb':lower, 'ub':upper}
        # print(name, percent)
      if percentile:
        group, lower, upper, counts = clip_percentile(group, column, upper_percentile=0.99)
        percentile_lt[name] = {'lb':lower, 'ub':upper}
      if mod_zscore:
        group, lower, upper, counts = clip_mod_zscore(group, column)
        mod_zscore_lt[name] = {'lb':lower, 'ub':upper}
      
      total_counts+=counts[0]
      lower_counts+=counts[1]
      upper_counts+=counts[2]
      scaled_group, scaler, mean, sd = minmax_scale_group(group, column)
      grouped_scalers[name] = {'scaler':scaler, 'mean':mean, 'sd':sd}         
      grouped_scaled_dfs.append(scaled_group)
  grouped_df = pd.concat(grouped_scaled_dfs)
  return grouped_df, grouped_scalers, (total_counts, lower_counts, upper_counts), (iqr_bds, zscore_bds, percentile_lt, mod_zscore_lt)




# %% [markdown]
# # 5. One Hot Encoding and scaling

# %%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def one_hot(df, encoder=None):
  if encoder is None:
    encoder = OneHotEncoder(sparse_output=False, 
                            handle_unknown="ignore"
                            )
    encoder.fit(df)
  encoded_data = encoder.transform(df)
  encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
  return encoded_df, encoder

def scale(df, scaler=None):
  if scaler is None:
    scaler = MinMaxScaler()

    #scaler = StandardScaler()
    scaler.fit(df)
  scaled_data = scaler.transform(df)
  scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
  return scaled_df, scaler

# %% [markdown]
# # Execute data preprocessing 

# %%
# data_path_0 = r"C:\Users\lc5744086\FIS AI Squad\TRAC-DEV\Copy of RCC22052025.xlsx"# Duplicate of Rcc1
data_path = r"Lenovo_Data.xlsx" 
df = pd.read_excel(data_path, header=0)
columns_to_select=categorical_columns+numeric_columns
data= df[columns_to_select]
data = data_sanity_check(data)
data,numeric_columns,categorical_columns=feature_engg(data,numeric_columns,categorical_columns)


# %% [markdown]
# # Data Grouping 

# %%
cpty_groups = {}
for cp in data['Cpty'].unique().tolist():
    buy_unique = data[data['Cpty']==cp]['BuyCurr'].unique().tolist()
    sell_unique = data[data['Cpty']==cp]['SellCurr'].unique().tolist()
    cpty_groups[cp] = {'buy': buy_unique, 'sell': sell_unique}
    # break

# %%
grouped_df, fv_scalers, total_counts, limits = group_points(data, groupby_columns_names, 'FaceValue',iqr=True)#facevalue
grouped_df, tdays_scalers, total_counts, ranges = group_points(data, group_trantype, 'TDays', percentile=True)

cat_data, ohe = one_hot(grouped_df[categorical_columns])
num_data, mms = scale(grouped_df[numeric_columns])
features = pd.concat([cat_data, num_data,], axis=1)
features['FaceValue'].fillna(0,inplace=True)

# %% [markdown]
# # 6. Save scaling Pickle 

# %%
import pickle
client_name = 'LENOVO'
models_path = r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\backend\pickle_folder"

pickle.dump({client_name: {'mms':mms, 'ohe':ohe, 'grouped_scalers': fv_scalers, 'cpty_group': cpty_groups, 'tdays_scalers': tdays_scalers}}, open(os.path.join(models_path, client_name+"_all_scales.pkl"), 'wb'))



# %% [markdown]
# # Combining data in list for Autoencoder

# %%
# finalize numeric and categorical feature index
catg_index_range=[0]
for col in (categorical_columns):
    catg_index_range.append(data[col].nunique())

feature_ohe_cat_index=[]
feature_ohe_all_index=[]
for i in range(1, len(catg_index_range)+1):
    feature_ohe_cat_index.append(sum(catg_index_range[0:i]))
feature_ohe_all_index = feature_ohe_cat_index.copy()
feature_ohe_all_index.append(feature_ohe_all_index[-1]+len(numeric_columns)+1)
del feature_ohe_cat_index

# %%
input_dict = dict()
for i in range(len(categorical_columns)+1):
    input_dict[f"input_{i}"] = features.iloc[:,feature_ohe_all_index[i]:feature_ohe_all_index[i+1]]

# %%
input_data =[]
for _, value in input_dict.items():
    input_data.append(value)

# %% [markdown]
# # *** Encoder model ***

# %% [markdown]
# # Helper functions for Tuning AutoEncoder

# %%
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
    


    @tf.function
    def range_weighted_mae(y_true, y_pred,scale=[1.00000000e+02, 1.00000000e+02, 1.00000000e+02, 1.00000000e+02,
       1.00000000e+02, 1.00000000e+02, 2.45994871e-11]):
        # convert back to raw units if you log+standardised ----------------
        #  log → expm1    and    z-score → *sigma + mu
        # but easier: just pass y_true_raw / y_pred_raw directly
        diff = tf.abs(y_pred - y_true) * scale   # (N, num_dim)
        return K.mean(diff, axis=-1)             # shape (N,)

    '''
    losses = (
    [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
    + [perc_mse]               # use for numeric head
    )
    '''
    losses = (
        [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
        + [keras.losses.MeanSquaredError()]
    )
    '''
    losses = (
    [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
    + [range_weighted_mae]  )              # replaces MSE
    '''

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

#--------------------------------------------------------------------


def plot_sorted_errors(row_score, top_fraction=0.02, show_gaps=True):
    """
    row_score     1-D array of reconstruction errors (one per row)
    top_fraction  0.02 ⇒ top-2 %; 0.01 ⇒ top-1 %; etc.
    show_gaps     set False if you only want the main curve
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


def elbow_anomaly_bundle_0(row_score,
                         blocks,
                         show_gaps=True,
                         loss_col="reconstruction_loss"):
    """
    Parameters
    ----------
    row_score : 1-D np.ndarray
    blocks    : list[pd.DataFrame] – original blocks (same row order)
    show_gaps : bool               – also plot Δ-error curve if True
    loss_col  : str                – name for loss column in df_anom

    Returns
    -------
    bundle       – [df_anom, df_all]
    elbow_score  – float
    elbow_idx    – int
    """
    # 1️⃣  elbow detection
    sorted_scores = np.sort(row_score)
    print('sorted_scores-->',sorted_scores)
    deltas        = np.diff(sorted_scores)
    print('deltas-->',deltas)
    elbow_idx     = np.argmax(deltas)
    print('elbow_idx-->',elbow_idx)
    elbow_score   = sorted_scores[elbow_idx]

    # 2️⃣  plots
    plt.figure(figsize=(7,4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx,  ls="--", color="tab:red",  label="elbow")
    plt.axhline(elbow_score,ls="--", color="tab:red")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7,3))
        plt.plot(deltas, lw=2)
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Gap between consecutive sorted errors")
        plt.xlabel("Index (i → i+1)"); plt.ylabel("Δ error")
        plt.tight_layout(); plt.show()

    # 3️⃣  build DataFrames
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} → {mask.sum():,} anomalies "
          f"({100*mask.mean():.2f} % of {len(row_score):,})")

    return [df_anom, df_all], elbow_score, elbow_idx

# ────────────────────────────────────────────────────────────────────────────
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
    print('---->',elbow_score)
    elbow_score=elbow_score
    anom_mask = row_score_train > elbow_score
    df_anom   = df_train_orig.loc[anom_mask].copy()
    df_anom[loss_col] = row_score_train[anom_mask]

    print(f"Anomalies in train split: {len(df_anom):,} "
          f"out of {len(df_train_orig):,} rows")

    return [df_anom, df_train_orig]

# ────────────────────────────────────────────────────────────────────────────
# Elbow detector with *relative-slope* threshold
#   • slope_frac    – how much steeper (e.g. 0.20 ⇒ +20 %) than the “flat”
#                     section a Δ-error must be to mark the elbow
#   • baseline_perc – first fraction of points used to define the flat slope
# ---------------------------------------------------------------------------
def elbow_anomaly_bundle(row_score,
                         blocks,
                         show_gaps=True,
                         loss_col="reconstruction_loss",
                         slope_frac=0.20,
                         baseline_perc=0.10):
    """
    Parameters
    ----------
    row_score     : 1-D array of reconstruction errors.
    blocks        : list[pd.DataFrame] – original blocks, still row-aligned.
    show_gaps     : bool – draw the Δ-error curve if True.
    loss_col      : str  – name of the error column in df_anom.
    slope_frac    : float (0–1) – elbow = first Δ exceeding
                    (1+slope_frac)*baseline_slope.
    baseline_perc : float (0–1) – left-hand share of the curve used
                    to estimate the baseline slope.

    Returns
    -------
    [df_anom, df_all] , elbow_score (float) , elbow_idx (int)
    """
    # 1️⃣  sort errors and compute consecutive differences
    sorted_scores = np.sort(row_score)
    deltas        = np.diff(sorted_scores)

    # 2️⃣  baseline slope from the flat left segment
    baseline_n    = max(1, int(len(deltas) * baseline_perc))
    baseline_slope= deltas[:baseline_n].mean()
    threshold     = (1.0 + slope_frac) * baseline_slope

    # 3️⃣  find the first Δ beyond threshold **after** the baseline zone
    search_idx    = np.arange(baseline_n, len(deltas))          # skip baseline part
    cand_idx      = search_idx[deltas[baseline_n:] > threshold]

    if cand_idx.size:                     # normal case
        elbow_idx = int(cand_idx[0])      # scalar
    else:                                 # fallback: largest overall jump
        elbow_idx = int(np.argmax(deltas))

    elbow_score = float(sorted_scores[elbow_idx])               # scalar

    # 4️⃣  diagnostic plots --------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx, ls="--", color="tab:red",
                label=f"elbow (>{slope_frac:.0%} jump)")
    plt.axhline(elbow_score, ls="--", color="tab:red")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7, 3))
        plt.plot(deltas, lw=2)
        plt.axhline(threshold, ls="--", color="tab:green",
                    label=f"threshold = {threshold:.3g}")
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Δ error between consecutive rows")
        plt.xlabel("Index (i→i+1)"); plt.ylabel("Δ error")
        plt.legend(); plt.tight_layout(); plt.show()

    # 5️⃣  build anomaly and full DataFrames --------------------------------
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} (idx {elbow_idx}) ⇒ "
          f"{mask.sum():,} anomalies ({100*mask.mean():.2f} % of "
          f"{len(row_score):,})")

    return [df_anom, df_all], elbow_score, elbow_idx



def elbow_anomaly_tail(row_score,
                       blocks,
                       max_fraction=0.05,      # stop if >5 % rows would be flagged
                       drop_ratio=0.20,        # 20 % relative drop defines elbow
                       loss_col="reconstruction_loss",
                       show_gaps=True):
    """
    • Works from the largest errors downwards.
    • Flags at most `max_fraction` of rows; could be fewer if a clear elbow appears earlier.
    """
    # 1️⃣ sort DESCENDING
    idx_sorted = np.argsort(row_score)[::-1]
    sorted_scores = row_score[idx_sorted]           # high → low
    deltas = -np.diff(sorted_scores)                # positive drops

    # 2️⃣ iterate until drop < drop_ratio OR max_fraction reached
    thresh_idx = int(np.ceil(max_fraction * len(row_score)))  # fallback
    base = sorted_scores[0]

    for i, gap in enumerate(deltas, start=1):
        if gap / base < drop_ratio:     # relative drop small → elbow
            thresh_idx = i
            break
        base = sorted_scores[i]

    elbow_score = sorted_scores[thresh_idx-1]       # last kept score

    # 3️⃣ build masks / DataFrames
    mask = row_score >= elbow_score
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    # 4️⃣ optional plots
    if show_gaps:
        plt.figure(figsize=(7,4))
        plt.plot(sorted_scores, lw=2)
        plt.axvline(thresh_idx-1, ls="--", color="tab:red",
                    label=f"elbow or {max_fraction:.0%} tail")
        plt.title("Descending Sorted Reconstruction Errors")
        plt.xlabel("Rank"); plt.ylabel("Error")
        plt.legend(); plt.tight_layout(); plt.show()

    print(f"Threshold @ rank {thresh_idx} (score {elbow_score:.6f}) "
          f"→ {mask.sum():,} anomalies "
          f"({100*mask.mean():.2f} % of {len(row_score):,})")

    return [df_anom, df_all], elbow_score, idx_sorted[:thresh_idx]


def find_jump_ratio(ratio_array,
                    max_fraction=0.05,
                    margin=0.02):
    """
    Returns the smallest ratio t such that the share of rows
    with r_i ≥ t is ≤ max_fraction.
    A small margin keeps you a bit below the hard cap.
    """
    sorted_r = np.sort(ratio_array)             # ascending
    N        = len(sorted_r)
    cut_idx  = int(np.ceil((1 - max_fraction) * N)) - 1
    # t is just above the value at cut_idx, then move down by margin
    t = sorted_r[cut_idx] * (1 + margin)
    return t


# ──────────────────────────────────────────────────────────────
# Helper: choose the smallest ratio that meets a target share
# ──────────────────────────────────────────────────────────────
def find_jump_ratio(ratio_array,             # r_i  = Δ / rolling-median
                    max_fraction=0.05,       # ≤ 5 % rows flagged
                    safety_margin=0.02):     # keep a bit under the cap
    """
    Returns a threshold τ so that
        share(rows where r_i >= T)  ≤ max_fraction.
    """
    sorted_r = np.sort(ratio_array)                    # ascending
    N         = len(sorted_r)
    cut_idx   = int(np.ceil((1 - max_fraction) * N)) - 1
    T         = sorted_r[cut_idx] * (1 + safety_margin)
    return T


# ──────────────────────────────────────────────────────────────
# Main: rolling-slope elbow detector with auto-ratio
# ──────────────────────────────────────────────────────────────
def elbow_anomaly_rolling(row_score,
                          blocks,
                          window_frac=0.02,     # window = 2 % of data
                          max_fraction=0.05,    # flag at most 5 %
                          loss_col="reconstruction_loss",
                          show_gaps=True):
    """
    Detect elbow from the right-hand tail using a rolling slope.
    • window_frac   – size of rolling window as fraction of dataset.
    • max_fraction  – hard cap on anomalies (e.g. 0.05 → 5 %).
    """
    # 1️⃣  sort errors (ascending) and compute Δ
    sorted_scores = np.sort(row_score)
    deltas        = np.diff(sorted_scores)

    # 2️⃣  rolling median of previous `window` slopes
    window = max(1, int(len(deltas) * window_frac))
    roll_med = np.zeros_like(deltas)
    for i in range(1, len(deltas)):
        lo = max(0, i - window)
        roll_med[i] = np.median(deltas[lo:i]) + 1e-12   # avoid /0

    # 3️⃣  ratio array for the search zone (skip first window)
    search_idx = np.arange(window, len(deltas))
    ratio_arr  = deltas[search_idx] / roll_med[search_idx]

    # 4️⃣  auto-select jump ratio to respect max_fraction
    jump_ratio = find_jump_ratio(ratio_arr, max_fraction=max_fraction)

    # 5️⃣  elbow index = first index where ratio ≥ jump_ratio
    cand       = search_idx[ratio_arr >= jump_ratio]
    elbow_idx  = int(cand[0]) if cand.size else len(sorted_scores) - 1
    elbow_score = float(sorted_scores[elbow_idx])

    # 6️⃣  plots ----------------------------------------------------
    plt.figure(figsize=(7,4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx, ls="--", color="tab:red",
                label=f"elbow (ratio ≥ {jump_ratio:.2f})")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7,3))
        plt.plot(deltas, lw=2, label="Δ error")
        thr_curve = np.where(np.arange(len(deltas))>=window,
                             roll_med*jump_ratio, np.nan)
        plt.plot(thr_curve, ls="--", color="tab:green",
                 label="dynamic threshold")
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Δ error vs dynamic threshold")
        plt.xlabel("Index (i→i+1)"); plt.ylabel("Δ error")
        plt.legend(); plt.tight_layout(); plt.show()

    # 7️⃣  build DataFrames ----------------------------------------
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} (idx {elbow_idx}) → "
          f"{mask.sum():,} anomalies "
          f"({100*mask.mean():.2f} % of {len(row_score):,})")

    return [df_anom, df_all], elbow_score, elbow_idx


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

# %%
# ----------------------------------------------------------------------
# Build reconstructed DataFrames and *then* inverse-transform
#   specified numeric columns with a single MinMaxScaler.
# ----------------------------------------------------------------------
def build_reconstructed_dataframes(blocks,
                                   recon,
                                   numeric_block_idx,
                                   numeric_cols: list,
                                   num_scaler,row_score,
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
    # --- inverse-transform *only* the requested numeric columns -------------
    '''
    for df_tmp in (df_recon_1hot, df_recon_prob,df_original):
        scaled_vals = df_tmp[numeric_cols].values          # (N, num_dim)
        raw_vals    = num_scaler.inverse_transform(scaled_vals)
        #if log:
        #    raw_vals =np.expm1(raw_vals)
        df_tmp[numeric_cols] = raw_vals                    # overwrite in place
    '''
    if save==True:
        df_recon_1hot.to_csv('scaled_prediction_onehot.csv')
        df_original.to_csv('scaled_original.csv')
        
    return df_recon_1hot, df_recon_prob,df_original

# %%
# BEFORE  re-train ---------------------------------------------------
raw_min   = input_data[6].min(0).values      # (num_dim,)
raw_max   = input_data[6].max(0).values
raw_range = np.maximum(raw_max - raw_min, 1)   # avoid /0
scale     = 100 / raw_range      # 1 % of range → value 1
scale

# %% [markdown]
# # Train Autoencoder

# %%
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
                                           patience=25,
                                           restore_best_weights=True)

history = model.fit(inputs_tr, targets_tr,                 # ← train data
                    validation_data=(inputs_va, targets_va),  # ← val data
                    epochs=150,
                    batch_size=128,
                    callbacks=[early_stop],
                    verbose=2)




# %% [markdown]
# # Save keras model

# %%
model.save("ad_model_inferance_LENOVO.keras") 

# %%
#original_train
# 5-C.  Reconstruction & anomaly scores -------------------------------------
cat_all, num_all, _, _ = prepare_blocks(blocks, numeric_block_idx)
inputs_all = cat_all + [num_all]          # len == 7 → matches model.inputs


recon = model.predict(inputs_all, batch_size=512, verbose=0)

# %%
# Compute errors
err_groups, row_score = compute_errors(cat_all, num_all, recon)

# after you have `row_score` and `blocks`
bundle, elbow_score, elbow_idx = elbow_anomaly_rolling(
        row_score=row_score,
        blocks=blocks,
        window_frac=0.95,
        show_gaps=False,
        max_fraction=0.20)

df_anom, df_all = bundle   # unpack as needed

plot_error_with_quantile_lines(row_score,quantiles=(0.01,0.03,0.05,0.10))

df_recon_1hot, df_recon_prob, df_original =build_reconstructed_dataframes(blocks,recon,numeric_block_idx=6,numeric_cols=numeric_columns[:-1],row_score=row_score,num_scaler=mms)

# --- saving happens OUTSIDE the function ---
#df_anom.to_csv("elbow_anomalies.csv", index=False)
#("Anomaly rows written to elbow_anomalies.csv")

# 5-D.  Plot learning curves -------------------------------------------------
block_names = [f"cat{i}" for i in range(len(cardinals))] + ["num"]
plot_history(history.history, block_names)

# %% [markdown]
# # Validate anomaly count 

# %%
counter_1 = 0
counter_2 = 0
counter_z = 0
non_anomaly = 0
thresh_one = 0.995
thresh_two = 0.95
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

# %%
# ─── 1.  elbow score ────────────────────────────────────────────────
sorted_scores = np.sort(row_score)
deltas        = np.diff(sorted_scores)
elbow_idx     = np.argmax(deltas)
elbow_score   = sorted_scores[elbow_idx]

print('--->',np.argsort(row_score))

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


# %% [markdown]
# # Testing with LLM

# %%
context = str({
"TranType" : "The type of FX transaction - FXSPOT, FXFORWARD or FXSWAP.",
"BuyCurr" : "The Buy currency in the deal.",
"SellCurr" : "The Sold currency in the deal.",
"BuyAmount" : "The Amount being bought and received in the currency of the buy currency.",
"SellAmount" : "The Amount being sold and paid in the currency of the sell currency.",
"SpotRate" : "The spot or base exchange rate the deal is being traded.",
"ForwardPoints" : "The difference between the spot rate and the forward rate in a foreign exchange contract.",
"PrimaryCurr" : "The main currency in the deal. This is also the nominated currency.",
"MaturityDate" :"The date the contract is considered complete.",
"Cpty" : "A counterparty is another organization that the client company deals with.  This can be an external organization or an internal branch or division of the client company.  In this instance, it is the party with whom the deal is being agreed to. This party is the legal entity being dealt with on the contract.",
"ActivationDate" : "The date at which the contract is started. This is the date which all properties of the contract are agreed and are binding.",
"BUnit" : "A business unit represents a distinct org entity within a client company, such as a division, department or a subsidiary.  Each business unit can operate semi-independently and is responsible for its own deal positions, exposures and financial activities.  It's also a concept that heps create separate tracking and reporting .",
"TDays" : "Transaction days is a derived feature from difference between Maturity date and Activation Date in terms of days for the deal",
"FaceValue" : "The Derived feature from Buy or Sell amount,which can be interpreted as the Deal amount converted in terms of Primary Currency.",
"Is_weekend_date" : "Confirms whether the deal happened on weekend dates. value '1' means dela happened on weekends.",
})

# %%
prompt ="""
### General Instructions:
- You are a helpful assistant specialized in fintech industry and also having some understanding of Machine Learning concpets.
- Your task is to give proper explanation for the Deal being anomaly using the 

### Input Data Desription:
- You will be provided with Deal data, i.e 'Data' in Json format, z_scores values of few Features in the 'Z_Score' object which is dictionary object followed by context in CSV format in form of 'context' object.
- The 'Data' object contains the actual Input values and the Difference values wrt the model outputs.
- The 'Z_Score' object will have z_scores values for the numerical features such as 'FaceValue' and 'TDays.
- The 'context' object will have the context on the features and their desricptions which should be helpful in explanantion for anomaly.

### Specific Instructions:
1. Your job is to analyze the "Difference" values of features from 'Data', 'z_score' values from 'Z_Score' objects respectively use the features with higher values for reasoning.
2. For z_scores consider features with values outside the range (-3, 3) for reasoning and ignore features within this range.
3. Always Look into the high deviated features your answer by presenting the 'z_score deviated features' and 'Top deviated Features' and then perform the analysis.
4.Always treat the 'BUnit' and 'Cpty' as single 'Business group' and this group owns the Deal data.
5.The explanation should always be Business group centric and other Features with highest deviations in Difference values or z_score values,
  should be clubbed with this group for the explanantion.
6.The explanation should start with "This FX deal with values Business unit '1', Counterparty 'A' appears anomalous ...", with the actual values of Business unit and Counter party being placed , followed by the features with high deviations.
7.In the Final explanation the feature names should be replaced with meaningful names by refering to 'context' object.
8.Provide your explanation in one or maximum Two sentences. Consider only top deviated features wrt z_scores and Difference values for explanation.



### **Expected Output format**:
You must return the Final Response in JSON format strictly for each input as follows:
{{
  \"Reason for Anomaly\": \"Reasoning that includes high deviated features and z_scores.\",
}}

### **Important Considerations** :
    - Always return the output as JSON object as per format mentioned above.No additional text.
    - Always ensure in the exlanation the short names should be replaced with Fullnames that can be derived from 'context' object.
    - The 'Reason for Anomaly' should not contain the Deviation values. It should only contain Feature names causing anomaly.
    - Always ensure the explanation should be 'Bunit' and 'Cpty' centric and the deviations in other features has to be summarized accrodingly.
    - Do not add any additional text after your final decision!
    - Always maintain professional tone while providing final explanation.

### Input data: 'Data','Z_Score' and 'context' object follows here.
You need to go through the 'Data' and follow all the above instructions and provide final response as per **Expected Output format**.
"""

# %%
#df_recon_1hot, df_recon_prob, df_original
'''
def get_llm_input(row):
  dfs = [df_original, df_recon_1hot, df_recon_prob]
  llm_data_input = pd.concat([df.iloc[row,:].to_frame() for df in dfs], axis=1)
  llm_data_input.columns = ['Input', 'Generated', 'Difference']
  llm_input = f'\nData:\n{llm_data_input.to_dict()}' + context
  return llm_input

llm_input=get_llm_input(17)
'''
def get_llm_output(llm_input):
  bot = Chatbot(FISLLM(), KnowledgeBases(), Chat.from_memory())
  bot.chat.add('system', prompt)
  return bot(llm_input)


def get_llm_input(row, context, features, reconstructed_features, zscores):
  dfs = [features, reconstructed_features, features-reconstructed_features]
  llm_data_input = pd.concat([df.iloc[row,:].to_frame() for df in dfs], axis=1)
  llm_data_input.columns = ['Actual', 'Expected', 'Difference']
  llm_input = f'\nData:\n{llm_data_input.to_dict()} \n' +f"\nZ_Scores:\n{zscores.to_dict(orient='records')}\n"+context
  return llm_input

# %%
def get_condition_filters(df_deviation):
  counter_1 = 0
  counter_2 = 0
  non_anomaly = 0
  thresh_one = 0.995
  thresh_two = 0.95
  Deviated_Features = []
  for idx, row in df_deviation.iterrows():
    filtered_columns = row[row > thresh_one].index.tolist()
    filtered_columns = [x for x in filtered_columns if not (x.startswith('BUnit') or x.startswith('Cpty'))]  
    filtered_columns_2 = row[row > thresh_two].index.tolist()
    filtered_columns_2 = [x for x in filtered_columns_2 if not (x.startswith('BUnit') or x.startswith('Cpty'))]
    if len(filtered_columns)>0:
        Deviated_Features.append({col: float(row[col]) for col in filtered_columns})
        counter_1+=1
    elif len(filtered_columns_2)>2:
        Deviated_Features.append({col: float(row[col]) for col in filtered_columns_2})
        counter_2+=1
    else:
        non_anomaly+=1
  return counter_1+counter_2 , non_anomaly, Deviated_Features

# %%
def get_filtered_data(features, reconstructed_df, zscores):
  df_deviation = features-reconstructed_df
  print (df_deviation)
  anomaly_count , non_anomaly_count, Deviated_Features = get_condition_filters(df_deviation)
  #print('Deviated_Features:', Deviated_Features)
  Z_score = zscores.to_dict(orient='records')[0]
  actual = [col for col in features.columns if (col.startswith('BUnit') or col.startswith('Cpty')) and (features[col] == 1).any()]
  Bunit = actual[0].split('_')
  Cpty = actual[1].split('_')
  filtered_data = {Bunit[0]:Bunit[1], Cpty[0]:Cpty[1], 'Z_score':Z_score, 'Deviated_Features':Deviated_Features}
  return filtered_data

# %%
def load_models(model_path, client_name='LENOVO'):
    model = keras.models.load_model(os.path.join(model_path, "ad_model_inferance_LENOVO.keras"))
    load_scalers = pickle.load(open(os.path.join(model_path, "all_scales_LENOVO.pkl"), 'rb'))[client_name]
    return model, load_scalers

model, load_scalers = load_models(models_path, client_name)

zs_facevalue = data.apply(lambda row: get_zscore(row['FaceValue'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['mean'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['sd']), axis=1)
zs_tdays = data.apply(lambda row: get_zscore(row['TDays'], load_scalers['tdays_scalers'][(row['Instrument'],)]['mean'], load_scalers['tdays_scalers'][(row['Instrument'],)]['sd']), axis=1)
zscores = pd.DataFrame({'Facevalue':zs_facevalue, 'TDays':zs_tdays})

# %%

#print(get_llm_output(llm_input)['answer'])

def get_zscore(x, mean, sd):
    epsilon=1e-9
    zscore = (x - mean)/(sd+epsilon)
    return zscore

# %%
def get_response(features, df_recon_1hot,zscores):
  response_list =[]
  anomalous_list = []
  result=1
  if type(result)==str:
    # test.loc[idx, ['Anomalous', 'Intensity', 'Explanation', 'DeviatedFeatures']] = ['Yes', "High", result, '']
    print(idx, result)
    response_list.append(result)
    anomalous_list.append("Yes")
  else:
    filtered_data = get_filtered_data(features, df_recon_1hot, zscores)
    # print("filtered_data:", filtered_data)
    if len(filtered_data['Deviated_Features'])>0:
      llm_input = str(filtered_data)
      outs = get_llm_output(llm_input)['answer']
      Anomalous = 'No'
      try:
        outs_dict = json.loads(outs)
        Explanation = outs_dict['Reason for Anomaly']
        Anomalous = 'Yes'
        # test.loc[idx, ['Anomalous', 'Intensity', 'Explanation', 'DeviatedFeatures']] = [Anomalous, Intensity, Explanation, deviation]
        print(outs_dict)
        anomalous_list.append(Anomalous)
        response_list.append(outs)
      except:
        print('@'*40)
        print(f'Index:{idx}',outs)
        print(f'Index:{idx}',type(outs))
        response_list.append(outs)
        Anomalous = "To be cheked in response"
        anomalous_list.append(Anomalous)
        # print('ERROR',result['RMSE'][0])
    else:
      Anomalous = 'No'
      anomalous_list.append(Anomalous)
      response_list.append('NA')
      print('This FX Deal apeears normal')
  try:
    response_df = pd.DataFrame({"Reason for Anomaly":response_list,"Anomaly":anomalous_list })
  except:
    pass
  return response_list,anomalous_list,response_df
#response_list,anomalous_list,response_df = get_response(row_list=list(range(15)), test_data = test3)

# %%
zscores.iloc[151932]

# %%
test_index=215637  

# %%
response_list,anomalous_list,response_df = get_response(df_original.iloc[[test_index]], df_recon_prob.iloc[[test_index]],zscores.iloc[[test_index]])

# %%
response_df


