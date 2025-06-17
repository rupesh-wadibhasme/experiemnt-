def scale(df, scaler=None):
  if scaler is None:
    #scaler = MinMaxScaler()
    
    scaler = StandardScaler()
    scaler.fit(df)
  scaled_data = scaler.transform(df)
  scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
  return scaled_df, scaler



from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# -----------------------------------------------------------------
# Forward scaling  (raw ➜ log1p ➜ StandardScaler)
# -----------------------------------------------------------------
def scale(df: pd.DataFrame, scaler: StandardScaler | None = None, log: bool = True):
    """
    df      : numeric DataFrame in original units
    scaler  : fitted StandardScaler or None (fit a new one)
    log     : apply np.log1p before scaling if True

    returns : scaled_df  (mean≈0, std≈1) ,  fitted scaler
    """
    x = np.log1p(df) if log else df
    if scaler is None:
        scaler = StandardScaler().fit(x)
    scaled_df = pd.DataFrame(scaler.transform(x), columns=df.columns).fillna(0)
    return scaled_df, scaler

import tensorflow.keras.backend as K

def perc_mse(y_true, y_pred):
    pct = (y_pred - y_true) / (K.maximum(K.abs(y_true), 1e-9))
    return K.mean(K.square(pct), axis=-1)

losses = (
    [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
    + [perc_mse]               # use for numeric head
)
model.compile(optimizer="adam", loss=losses)

# -----------------------------------------------------------------
# Inverse scaling  (scaled ➜ un-scale ➜ expm1)
# -----------------------------------------------------------------
def inverse_scale(arr, scaler: StandardScaler, columns, log: bool = True):
    """
    arr      : ndarray or DataFrame in scaled space
    scaler   : the StandardScaler returned by `scale`
    columns  : column names (preserves order)
    log      : True if log1p was used during scaling
    """
    raw = scaler.inverse_transform(arr)
    if log:
        raw = np.expm1(raw)
    return pd.DataFrame(raw, columns=columns)
