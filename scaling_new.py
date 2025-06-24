# BEFORE  re-train ---------------------------------------------------
raw_min   = numeric_df.min(0).values      # (num_dim,)
raw_max   = numeric_df.max(0).values
raw_range = np.maximum(raw_max - raw_min, 1)   # avoid /0
scale     = 100 / raw_range      # 1 % of range → value 1

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.function
def range_weighted_mae(y_true, y_pred):
    # convert back to raw units if you log+standardised ----------------
    #  log → expm1    and    z-score → *sigma + mu
    # but easier: just pass y_true_raw / y_pred_raw directly
    diff = tf.abs(y_pred - y_true) * scale   # (N, num_dim)
    return K.mean(diff, axis=-1)             # shape (N,)


losses = (
    [keras.losses.SparseCategoricalCrossentropy()] * len(cardinals)
    + [range_weighted_mae]                # replaces MSE
)
model.compile(optimizer="adam", loss=losses)
