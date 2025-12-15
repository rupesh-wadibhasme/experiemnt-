
# Import required libraries
import os
import re
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from pandas.tseries.offsets import BDay
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Dense, Embedding, Flatten, Concatenate, 
                                     BatchNormalization, LayerNormalization, Dropout,
                                     MultiHeadAttention, Add, Activation, Reshape)
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore')

def _seed():
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_colwidth", None)
_seed()

# Load dataset
file_path = r"C:\Users\lc5753474\Documents\bank_anomaly\Data\DL_FACT_BANKTRANSACTION_2023_Broadcom.xlsx"  # Update path if needed
try:
    df_full = pd.read_excel(file_path, usecols=['BankTransactionId', 'BankAccountCode', 'BankTransactionCode', 'BusinessUnitCode', 'ValueDateKey', 'PostingDateKey', 'AmountInBankAccountCurrency'])
    print(f"Loaded {len(df_full)} rows.")
except Exception as e:
    print(f"Error loading Excel file: {e}")
    df_full = pd.DataFrame()
df_full.shape

import gc
gc.collect()

def pre_processing(dataF: pd.DataFrame) -> pd.DataFrame:
  # Handle missing values and parse dates
  important_features = ['ValueDateKey', 'PostingDateKey', 'BankTransactionId', 'BankAccountCode', 'BusinessUnitCode', 'BankTransactionCode']
  for col in important_features:
      if col not in dataF.columns:
          print(f"Warning: {col} not found in dataset.")
  dataF = dataF.dropna(subset=important_features)
  for date_col in ['ValueDateKey', 'PostingDateKey']:
      if date_col in dataF.columns:
          dataF[date_col] = pd.to_datetime(dataF[date_col], format="%Y%m%d", errors='coerce')
  dataF = dataF.dropna(subset=['ValueDateKey', 'PostingDateKey'])
  dataF.reset_index(drop=True, inplace=True)
  return dataF

# Feature engineering: Enhanced with temporal, amount-based, and group statistics features

# 1. Temporal features
def is_business_day(date):
    return pd.notnull(date) and pd.Timestamp(date).dayofweek < 5

def feature_engineering(dataF: pd.DataFrame) -> pd.DataFrame:
  # 1. Temporal features
  dataF['IsBusinessDay_Value'] = dataF['ValueDateKey'].apply(is_business_day)
  dataF['IsBusinessDay_Posting'] = dataF['PostingDateKey'].apply(is_business_day)
  # dataF['DayOfWeek_Value'] = dataF['ValueDateKey'].dt.dayofweek
  dataF['Month_Value'] = dataF['ValueDateKey'].dt.month
  # dataF['Quarter_Value'] = dataF['ValueDateKey'].dt.quarter
  # dataF['DaysBetweenPostingValue'] = (dataF['PostingDateKey'] - dataF['ValueDateKey']).dt.days

  # 2. Group-based features
  # group_cols = ['BankAccountCode','BankTransactionCode','BusinessUnitCode','ValueDateKey']
  dataF['TransCountPerAccountDay'] = dataF.groupby(group_cols)['BankTransactionId'].transform('count')
  dataF['GroupSize'] = dataF.groupby(group_cols)['BankTransactionId'].transform('size')
  # df['ReferenceLength'] = df['BankTransactionId'].astype(str).apply(len)
  def identical_amounts(x):
      return x.nunique() == 1 and len(x) > 1
  dataF['IdenticalAmountsInGroup'] = dataF.groupby(group_cols)['AmountInBankAccountCurrency'].transform(identical_amounts)
  dataF['AmountDeviation'] = dataF.groupby(group_cols)['AmountInBankAccountCurrency'].transform(lambda x: abs(x - x.mean()) if len(x) > 1 else np.nan)
  dataF['SingleTransactionGroup'] = dataF['GroupSize'] == 1

  # 3. Advanced amount-based features
  # Group statistics (calculated with signed amounts first)
  dataF['GroupAmountMean'] = dataF.groupby(group_cols)['AmountInBankAccountCurrency'].transform('mean')
  # dataF['GroupAmountStd'] = dataF.groupby(group_cols)['AmountInBankAccountCurrency'].transform('std').fillna(0)
  dataF['AmountRatioToGroupMean'] = dataF['AmountInBankAccountCurrency'] / (dataF['GroupAmountMean'] + 1e-8)
  
  # Amount percentile within group (signed)
  # dataF['AmountPercentileInGroup'] = dataF.groupby(group_cols)['AmountInBankAccountCurrency'].transform(
    # lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
  # )
  
  # Now create absolute amount and sign features
  dataF['AmountSign'] = (dataF['AmountInBankAccountCurrency'] >= 0).astype(int)  # 1 for positive, 0 for negative
  dataF['AbsAmount'] = np.abs(dataF['AmountInBankAccountCurrency'])  # Absolute amount
  dataF['LogAbsAmount'] = np.log1p(dataF['AbsAmount'])  # Log transform for large values

  # 4. Cross-feature interactions
  # dataF['AmountTransCountInteraction'] = dataF['AbsAmount'] * dataF['TransCountPerAccountDay']  # Use absolute amount for interaction

  # Fill missing values in derived features
  dataF.fillna(0, inplace=True)
  print(f"\nFeature engineering complete. Total features: {len(dataF.columns)}")
  return dataF

def nomalize_data(dataF: pd.DataFrame) -> pd.DataFrame:
  # Prepare features with improved scaling strategies
  # Categorical encoding
  cat_indices = {}
  for col in cat_features:
      if col in dataF.columns:
          dataF[col] = dataF[col].astype('category')
          cat_indices[col] = dict(enumerate(dataF[col].cat.categories))
          dataF[col] = dataF[col].cat.codes

  # Amount features - use RobustScaler for better outlier handling
  # robust_scaler = RobustScaler()
  # dataF[amount_features] = robust_scaler.fit_transform(dataF[amount_features])
  scale_features = amount_features + count_features + temporal_features
  # Count and deviation features - use StandardScaler
  standard_scaler = StandardScaler()
  dataF[scale_features] = standard_scaler.fit_transform(dataF[scale_features])

  # Convert binary features to float
  dataF[binary_features] = dataF[binary_features].astype(float)

  # Combine all numeric features
  # scale_features = amount_features + count_features
  num_features = scale_features + binary_features # + temporal_features
  total_features = cat_features + num_features

  print(f"\nFeature categories:")
  print(f"  Categorical: {len(cat_features)}")
  print(f"  Numeric (scaled): {len(scale_features)}")
  print(f"  Binary: {len(binary_features)}")
  print(f"  Temporal: {len(temporal_features)}")
  print(f"  Total features: {len(total_features)}")
  return dataF, cat_indices, total_features, num_features,  standard_scaler

# Define a sequential autoencoder model using Keras.


# Enhanced autoencoder with attention mechanism and normalization
def build_autoencoder_with_attention(dataF: pd.DataFrame, cat_features: list, num_features: list) -> Model:
  """
  Build an enhanced autoencoder with:
  - Multi-head attention mechanism
  - Batch and layer normalization
  - Dropout for regularization
  - Residual connections
  """
  inputs = []
  embeddings = []
  output_layers = []
  
  # Embedding layers for categorical features
  cat_feature_dims = {}
  for col in cat_features:
      num_categories = dataF[col].nunique()
      cat_feature_dims[col] = num_categories
      input_cat = Input(shape=(1,), name=f'{col}_input')
      embed_dim = min(50, (num_categories + 1) // 2)
      embed = Embedding(input_dim=num_categories+1, output_dim=embed_dim, name=f'{col}_embed')(input_cat)
      embed = Flatten()(embed)
      # Add normalization to embeddings
      embed = LayerNormalization(name=f'{col}_ln')(embed)
      inputs.append(input_cat)
      embeddings.append(embed)
      # Output layer for categorical feature
      output_layers.append(Dense(num_categories, activation='softmax', name=f'{col}_output'))
  
  # Numerical features with normalization
  input_num = Input(shape=(len(num_features),), name='num_input')
  inputs.append(input_num)
  num_normalized = LayerNormalization(name='num_ln')(input_num)
  embeddings.append(num_normalized)
  output_layers.append(Dense(len(num_features), activation='linear', name='num_output'))
  
  # Concatenate all features
  x = Concatenate(name='concat_features')(embeddings)
  concat_dim = x.shape[1]
  
  # === ENCODER with Attention ===
  # First encoding layer with normalization
  encoding_dim_1 = max(64, concat_dim // 2)
  encoded_1 = Dense(encoding_dim_1, name='encoder_dense_1')(x)
  encoded_1 = BatchNormalization(name='encoder_bn_1')(encoded_1)
  encoded_1 = Activation('relu', name='encoder_act_1')(encoded_1)
  encoded_1 = Dropout(0.2, name='encoder_dropout_1')(encoded_1)
  
  # Second encoding layer
  encoding_dim_2 = max(32, encoding_dim_1 // 2)
  encoded_2 = Dense(encoding_dim_2, name='encoder_dense_2')(encoded_1)
  encoded_2 = BatchNormalization(name='encoder_bn_2')(encoded_2)
  encoded_2 = Activation('relu', name='encoder_act_2')(encoded_2)
  encoded_2 = Dropout(0.2, name='encoder_dropout_2')(encoded_2)
  
  # Bottleneck layer
  encoding_dim_bottleneck = max(16, encoding_dim_2 // 2)
  encoded_bottleneck = Dense(encoding_dim_bottleneck, name='bottleneck_dense')(encoded_2)
  encoded_bottleneck = BatchNormalization(name='bottleneck_bn')(encoded_bottleneck)
  encoded_bottleneck = Activation('relu', name='bottleneck_act')(encoded_bottleneck)
  
  # === MULTI-HEAD ATTENTION ===
  # Reshape for attention: (batch, 1, features) -> treating features as sequence
  # attention_input = tf.expand_dims(encoded_bottleneck, axis=1)
  attention_input = Reshape((1, encoding_dim_bottleneck), name='reshape_for_attention')(encoded_bottleneck)
  
  # Multi-head attention layer
  num_heads = min(4, encoding_dim_bottleneck // 4) if encoding_dim_bottleneck >= 4 else 1
  attention_output = MultiHeadAttention(
      num_heads=num_heads, 
      key_dim=encoding_dim_bottleneck // num_heads if num_heads > 0 else encoding_dim_bottleneck,
      dropout=0.1,
      name='multi_head_attention'
  )(attention_input, attention_input)
  
  # Flatten attention output
  attention_output = Flatten(name='attention_flatten')(attention_output)
  
  # Residual connection + layer normalization
  encoded_with_attention = Add(name='residual_add')([encoded_bottleneck, attention_output])
  encoded_with_attention = LayerNormalization(name='attention_ln')(encoded_with_attention)
  
  # === DECODER with Normalization ===
  # First decoder layer
  decoded_1 = Dense(encoding_dim_2, name='decoder_dense_1')(encoded_with_attention)
  decoded_1 = BatchNormalization(name='decoder_bn_1')(decoded_1)
  decoded_1 = Activation('relu', name='decoder_act_1')(decoded_1)
  decoded_1 = Dropout(0.2, name='decoder_dropout_1')(decoded_1)
  
  # Second decoder layer
  decoded_2 = Dense(encoding_dim_1, name='decoder_dense_2')(decoded_1)
  decoded_2 = BatchNormalization(name='decoder_bn_2')(decoded_2)
  decoded_2 = Activation('relu', name='decoder_act_2')(decoded_2)
  decoded_2 = Dropout(0.2, name='decoder_dropout_2')(decoded_2)
  
  # Final decoder layer to match concatenated input dimension
  decoded_final = Dense(concat_dim, name='decoder_dense_final')(decoded_2)
  decoded_final = BatchNormalization(name='decoder_bn_final')(decoded_final)
  decoded_final = Activation('relu', name='decoder_act_final')(decoded_final)
  
  # === OUTPUT LAYERS ===
  outputs = [out(decoded_final) for out in output_layers]
  
  # Build model
  autoencoder = Model(inputs=inputs, outputs=outputs, name='autoencoder_with_attention')
  return autoencoder

# Keep original function for backward compatibility
def build_autoencoder(dataF: pd.DataFrame, cat_features: list, num_features: list) -> Model:
  """Original simple autoencoder (kept for reference)"""
  inputs = []
  embeddings = []
  output_layers = []
  cat_feature_dims = {}
  for col in cat_features:
      num_categories = dataF[col].nunique()
      cat_feature_dims[col] = num_categories
      input_cat = Input(shape=(1,), name=f'{col}_input')
      embed_dim = min(50, (num_categories + 1) // 2)
      embed = Embedding(input_dim=num_categories+1, output_dim=embed_dim, name=f'{col}_embed')(input_cat)
      embed = Flatten()(embed)
      inputs.append(input_cat)
      embeddings.append(embed)
      output_layers.append(Dense(num_categories, activation='softmax', name=f'{col}_output'))
  input_num = Input(shape=(len(num_features),), name='num_input')
  inputs.append(input_num)
  embeddings.append(input_num)
  output_layers.append(Dense(len(num_features), activation='linear', name='num_output'))
  x = Concatenate()(embeddings)
  encoding_dim = max(4, x.shape[1] // 2)
  encoded = Dense(encoding_dim, activation='relu')(x)
  outputs = [out(encoded) for out in output_layers]
  autoencoder = Model(inputs=inputs, outputs=outputs)
  return autoencoder

def prepare_dataset(dataF: pd.DataFrame, cat_features: list, num_features: list):
  # Prepare input data for model
  X_cat = [dataF[col].values for col in cat_features]
  X_num = dataF[num_features].values
  X_all = X_cat + [X_num]
  # Prepare targets for training (integer for sparse categorical)
  y_cat_sparse = [dataF[col].values for col in cat_features]
  y_num = X_num
  y_all_sparse = y_cat_sparse + [y_num]
  return X_all, y_all_sparse, X_cat, X_num


df = df_full.copy()
df.shape

group_cols = ['BankAccountCode','BankTransactionCode','BusinessUnitCode','ValueDateKey']
cat_features = ['BankAccountCode', 'BusinessUnitCode', 'BankTransactionCode']

binary_features = ['IsBusinessDay_Value', 'IsBusinessDay_Posting', 'SingleTransactionGroup', 
                  'IdenticalAmountsInGroup', 'AmountSign']

# amount_features = ['AbsAmount', 'LogAbsAmount', 'AmountRatioToGroupMean', 
#                     'AmountTransCountInteraction', 'GroupAmountMean', 'GroupAmountStd']
# # Temporal features (already encoded as integers)
# temporal_features = ['DayOfWeek_Value', 'Month_Value', 'Quarter_Value']
# count_features = ['TransCountPerAccountDay', 'AmountDeviation', 
#                     'AmountPercentileInGroup', 'DaysBetweenPostingValue'] #GroupSize
amount_features = ['AbsAmount', 'LogAbsAmount', 'AmountRatioToGroupMean']
# Temporal features (already encoded as integers)
temporal_features = ['Month_Value']
count_features = ['TransCountPerAccountDay', 'AmountDeviation']
                    # 'AmountPercentileInGroup', 'DaysBetweenPostingValue'] #GroupSize

df = pre_processing(df)
df = feature_engineering(df)
df, cat_indices, total_features, num_features, scaler = nomalize_data(df)
X_all, y_all_sparse, X_cat, X_num = prepare_dataset(df, list(cat_indices.keys()), num_features)

# Use the enhanced autoencoder with attention
print("Building enhanced autoencoder with attention mechanism and normalization...")
autoencoder = build_autoencoder_with_attention(df, list(cat_indices.keys()), num_features)

# Compile with same losses but add metrics for monitoring
losses = ['sparse_categorical_crossentropy'] * len(list(cat_indices.keys())) + ['mse']
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=losses,
    metrics=['accuracy'] * len(list(cat_indices.keys())) + ['mae']
)

autoencoder.summary()

# Train the autoencoder using the fit() method and visualize training progress.

# Train the autoencoder with improved hyperparameters
# Callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)

history = autoencoder.fit(
    X_all, y_all_sparse, 
    epochs=50,  # Increased epochs with early stopping
    batch_size=64,  # Larger batch for better generalization
    validation_split=0.2, 
    verbose=1, 
    callbacks=[early_stop, reduce_lr]
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Autoencoder Training Loss')
plt.grid(True)
plt.show()

def save():
  model_path = os.path.join(os.getcwd(),str(datetime.datetime.now()), "models")
  os.makedirs(model_path, exist_ok=True)
  # --- Save the autoencoder model and scalers ---
  autoencoder.save(os.path.join(model_path,'bank_autoencoder_model.keras'))  # Save in Keras format
  print("Model saved as 'bank_autoencoder_model.keras'")

  scalers = {
      'cat_indices': cat_indices, 
      'robust_scaler': scaler[0],
      'standard_scaler': scaler[1],
  }
  with open(os.path.join(model_path,'scalers.pkl'), 'wb') as f:
      pickle.dump(scalers, f)
  print("Scalers saved as 'scalers.pkl'")

# --- Load the autoencoder model ---
def _load():
  autoencoder = load_model(r'C:\Users\lc5753474\Documents\bank_anomaly\bank_autoencoder_model.keras')
  print("Model loaded from 'bank_autoencoder_model.keras'")

  with open(r'C:\Users\lc5753474\Documents\bank_anomaly\scalers.pkl', 'rb') as f_scaler:
      loaded_scaler = pickle.load(f_scaler)
  cat_indices = loaded_scaler['cat_indices']
  scaler = loaded_scaler['num_scaler']
  print("scaler loaded from 'scalers.pkl'")


# Use reconstruction error to flag anomalies in the dataset.

scale_features = amount_features + count_features + temporal_features
# num_features = scale_features + binary_features + temporal_features


# Predict outputs
X_pred = autoencoder.predict(X_all)
cat_preds = X_pred[:-1]  # List of categorical outputs (softmax)
num_pred = X_pred[-1]    # Numeric output

# Numeric predictions
# num_pred_df = pd.DataFrame(scaler.inverse_transform(num_pred[:, :len(amount_features)]), columns=amount_features, index=df.index)
# num_pred_df[count_features] = scaler[1].inverse_transform(num_pred[:, len(amount_features):len(amount_features)+len(count_features)])
# num_pred_df[binary_features] = (num_pred[:, len(amount_features)+len(count_features):len(amount_features)+len(count_features)+len(binary_features)] > 0.5).astype(float)
# num_pred_df[temporal_features] = num_pred[:, len(amount_features)+len(count_features)+len(binary_features):]
num_pred_df = pd.DataFrame(scaler.inverse_transform(num_pred[:, :len(scale_features)]), columns=scale_features, index=df.index)
num_pred_df[binary_features] = (num_pred[:, len(scale_features):len(scale_features)+len(binary_features)] > 0.5).astype(float)
# Categorical predictions
cat_pred_dfs = []
for i, cat_pred in enumerate(cat_preds):
    pred_codes = np.argmax(cat_pred, axis=1)
    pred_values = [cat_indices[cat_features[i]].get(code, code) for code in pred_codes]
    cat_pred_dfs.append(pd.Series(pred_values, name=cat_features[i], index=df.index))
cat_pred_df = pd.concat(cat_pred_dfs, axis=1)

# Combine all predictions
pred_df = pd.concat([cat_pred_df, num_pred_df], axis=1)


# Numeric reconstruction error (MSE)
mse = np.mean(np.power(X_num - num_pred, 2), axis=1)
num_feature_errors = np.abs(X_num - num_pred)  # shape: (n_samples, n_num_features)

# Categorical reconstruction error (cross-entropy)
cat_errors = []
for i, cat_pred in enumerate(cat_preds):
    # True labels for this categorical feature
    true_labels = X_cat[i]
    # Cross-entropy error for each sample
    ce = -np.log(cat_pred[np.arange(len(cat_pred)), true_labels] + 1e-8)
    cat_errors.append(ce)
cat_errors = np.stack(cat_errors, axis=1)
cat_error_mean = np.mean(cat_errors, axis=1)

# Combine errors (simple sum or weighted sum)
combined_error = mse + cat_error_mean  # You can adjust weights if needed
df['reconstruction_error'] = combined_error

# ============= Multiple Threshold Methods =============
print("\n" + "="*80)
print("ANOMALY THRESHOLD ANALYSIS - Multiple Methods")
print("="*80)

# Method 1: Statistical approach (IQR - Interquartile Range)
Q1 = np.percentile(combined_error, 25)
Q3 = np.percentile(combined_error, 75)
IQR = Q3 - Q1
threshold_iqr = Q3 + 1.5 * IQR  # Standard outlier detection
anomalies_iqr = combined_error > threshold_iqr

# Method 2: Z-score approach (3 standard deviations)
mean_error = np.mean(combined_error)
std_error = np.std(combined_error)
threshold_zscore = mean_error + 3 * std_error  # 3-sigma rule
anomalies_zscore = combined_error > threshold_zscore

# Method 3: Modified Z-score (robust to outliers using MAD)
median_error = np.median(combined_error)
mad = np.median(np.abs(combined_error - median_error))
modified_zscore = 0.6745 * (combined_error - median_error) / (mad + 1e-8)
threshold_modified_z = median_error + 3.5 * (mad / 0.6745)  # 3.5 is common threshold
anomalies_modified_z = modified_zscore > 3.5

# Method 4: Percentile-based (adjustable contamination)
contamination_rate = 0.01  # 1% expected anomalies (adjust based on domain knowledge)
threshold_percentile = np.percentile(combined_error, (1 - contamination_rate) * 100)
anomalies_percentile = combined_error > threshold_percentile

# Method 5: Elbow method - find the "knee" in sorted errors
sorted_errors = np.sort(combined_error)
n = len(sorted_errors)
# Calculate rate of change
diffs = np.diff(sorted_errors)
# Find the largest jump
knee_idx = np.argmax(diffs) + 1
threshold_elbow = sorted_errors[knee_idx]
anomalies_elbow = combined_error > threshold_elbow

# Compare all methods
print(f"\n{'Method':<25} {'Threshold':<15} {'Anomalies':<15} {'% of Data':<15}")
print(f"{'-'*70}")
print(f"{'1. IQR (1.5×IQR)':<25} {threshold_iqr:<15.4f} {anomalies_iqr.sum():<15} {(anomalies_iqr.sum()/len(df)*100):.2f}%")
print(f"{'2. Z-score (3σ)':<25} {threshold_zscore:<15.4f} {anomalies_zscore.sum():<15} {(anomalies_zscore.sum()/len(df)*100):.2f}%")
print(f"{'3. Modified Z-score':<25} {threshold_modified_z:<15.4f} {anomalies_modified_z.sum():<15} {(anomalies_modified_z.sum()/len(df)*100):.2f}%")
print(f"{'4. Percentile (99%)':<25} {threshold_percentile:<15.4f} {anomalies_percentile.sum():<15} {(anomalies_percentile.sum()/len(df)*100):.2f}%")
print(f"{'5. Elbow Method':<25} {threshold_elbow:<15.4f} {anomalies_elbow.sum():<15} {(anomalies_elbow.sum()/len(df)*100):.2f}%")

# Recommendation: Use IQR or Modified Z-score (most robust)
print(f"\n" + "="*80)
print("RECOMMENDATION:")
print("  • IQR Method: Best for general use, robust to outliers")
print("  • Modified Z-score: Best when data is highly skewed")
print("  • Percentile: Use when you know expected anomaly rate")
print("="*80)






