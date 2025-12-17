# Databricks notebook source
!pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from pathlib import Path
from config.config import *
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import AutoencoderModel
from src.utils import (
    normalize_data_train, prepare_dataset,
    calculate_reconstruction_error, compute_threshold,
    save_artifacts, plot_training_history, plot_reconstruction_errors
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
mlflow.set_tracking_uri('databricks')

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)


# COMMAND ----------

TRAIN_FILES = ['/Volumes/aimluat/treasury_forecasting/broadcom/DL_FACT_BANKTRANSACTION_2022_Broadcom.xlsx', '/Workspace/Users/lc5753474@fisprodcloud.com/bank_anomaly/DL_FACT_BANKTRANSACTION_2023_Broadcom.xlsx']

# COMMAND ----------

# Set random seed
set_seed(RANDOM_SEED)
print(f"\n✓ Random seed set to {RANDOM_SEED}")
# LOAD AND PREPROCESS DATA
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

preprocessor = DataPreprocessor()
df = preprocessor.load_and_preprocess(TRAIN_FILES)
preprocessor.print_data_summary(df)

# COMMAND ----------

# TRAIN/VALIDATION SPLIT
print("\n" + "="*80)
print("STEP 2: TRAIN/VALIDATION SPLIT")
print("="*80)

# Split by date
df_train = df[df['ValueDateKey'] < datetime.strptime(str(TRAIN_END_DATE), "%Y%m%d")].copy()
df_val = df[df['ValueDateKey'] >= datetime.strptime(str(TRAIN_END_DATE), "%Y%m%d")].copy()

print(f"\nSplit Configuration:")
print(f"  Train: Before {TRAIN_END_DATE}")
print(f"  Val:   From {TRAIN_END_DATE}")
print(f"\nDataset Sizes:")
print(f"  Train: {len(df_train):,} records ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(df_val):,} records ({len(df_val)/len(df)*100:.1f}%)")

# COMMAND ----------

# FEATURE ENGINEERING
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

engineer = FeatureEngineer()

# Compute training statistics and create features
df_train, train_stats = engineer.engineer_features(df_train, is_training=True)

# Apply validation features using training statistics (WITH CLIPPING)
df_val, _ = engineer.engineer_features(
df_val, 
is_training=False, 
train_stats=train_stats,
clip_data=True  # Clip validation data for consistent distribution
)

print(f"\nFeatures Created:")
print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
print(f"  Amount: {len(AMOUNT_FEATURES)}")
print(f"  Count: {len(COUNT_FEATURES)}")
print(f"  Temporal: {len(TEMPORAL_FEATURES)}")
print(f"  Binary: {len(BINARY_FEATURES)}")
print(f"  Total: {len(TOTAL_FEATURES)}")

# COMMAND ----------

# NORMALIZATION (ENCODING + SCALING)
print("\n" + "="*80)
print("STEP 4: NORMALIZATION (ENCODING + SCALING)")
print("="*80)

df_train, df_val, cat_indices, total_features, num_features, scalers = normalize_data_train(
df_train,
df_val,
CATEGORICAL_FEATURES,
SCALE_FEATURES,
BINARY_FEATURES
)

# PREPARE DATASETS FOR MODEL
print("\n" + "="*80)
print("STEP 5: PREPARING DATASETS FOR MODEL")
print("="*80)

X_train_all, y_train_all, X_train_cat, X_train_num = prepare_dataset(
df_train,
CATEGORICAL_FEATURES,
num_features
)

X_val_all, y_val_all, X_val_cat, X_val_num = prepare_dataset(
df_val,
CATEGORICAL_FEATURES,
num_features
)

print(f"\n✓ Training data prepared: {len(X_train_all[0]):,} samples")
print(f"✓ Validation data prepared: {len(X_val_all[0]):,} samples")
print(f"  Inputs: {len(X_train_all)} arrays (3 categorical + 1 numeric)")
print(f"  Outputs: {len(y_train_all)} arrays (3 categorical + 1 numeric)")

# COMMAND ----------

# BUILD MODEL
print("\n" + "="*80)
print("STEP 6: BUILDING AUTOENCODER MODEL")
print("="*80)

# Create and build the autoencoder model
autoencoder_model = AutoencoderModel(
cat_features=CATEGORICAL_FEATURES,
num_features=num_features
)

# Build and compile
autoencoder_model.build(df_train).compile()

print(f"\nModel Architecture Summary:")
print(f"  Encoder: {ENCODER_CONFIG['hidden_layers']} → {ENCODER_CONFIG['latent_dim']}")
print(f"  Decoder: {DECODER_CONFIG['hidden_layers']}")
print(f"  Attention: {ATTENTION_CONFIG['use_attention']} ({ATTENTION_CONFIG['num_heads']} heads)")
print(f"  Dropout: {ENCODER_CONFIG['dropout_rate']}")
print(f"  L2 Reg: {ENCODER_CONFIG['l2_regularization']}")

# COMMAND ----------

# TRAIN MODEL
print("\n" + "="*80)
print("STEP 7: TRAINING MODEL")
print("="*80)

history = autoencoder_model.train(
X_train_all, y_train_all,
X_val_all, y_val_all,
use_callbacks=True
)

# COMMAND ----------

# COMPUTE ANOMALY THRESHOLD
print("\n" + "="*80)
print("STEP 8: COMPUTING ANOMALY THRESHOLD")
print("="*80)

# Compute errors on combined train+val data for better threshold
print("\nComputing reconstruction errors on ALL data (train + val)...")
X_all_combined = [np.concatenate([X_train_all[i], X_val_all[i]]) for i in range(len(X_train_all))]
X_cat_combined = [np.concatenate([X_train_cat[i], X_val_cat[i]]) for i in range(len(X_train_cat))]
X_num_combined = np.concatenate([X_train_num, X_val_num])

combined_error, _, _, _ = calculate_reconstruction_error(
autoencoder_model.model,  # Use the underlying Keras model
X_all_combined,
X_cat_combined,
X_num_combined,
weighted=True
)

# Compute threshold using configured method
threshold = compute_threshold(
combined_error,
method=ANOMALY_DETECTION_CONFIG['threshold_method'],
factor=ANOMALY_DETECTION_CONFIG.get('iqr_factor', 1.5),
sigma=ANOMALY_DETECTION_CONFIG.get('zscore_sigma', 3),
threshold=ANOMALY_DETECTION_CONFIG.get('modified_z_threshold', 3.5),
contamination=ANOMALY_DETECTION_CONFIG.get('contamination_rate', 0.01)
)

n_anomalies = (combined_error > threshold).sum()
anomaly_pct = (n_anomalies / len(combined_error)) * 100

print(f"\n✓ Threshold computed using {ANOMALY_DETECTION_CONFIG['threshold_method'].upper()} method:")
print(f"  Threshold: {threshold:.4f}")
print(f"  Anomalies in training data: {n_anomalies:,} ({anomaly_pct:.2f}%)")
print(f"  Error range: [{combined_error.min():.4f}, {combined_error.max():.4f}]")
print(f"  Error mean: {combined_error.mean():.4f}")
print(f"  Error std: {combined_error.std():.4f}")

# COMMAND ----------

# SAVE MODEL AND ARTIFACTS
print("\n" + "="*80)
print("STEP 9: SAVING MODEL AND ARTIFACTS")
print("="*80)
model_path = get_model_path()
autoencoder_model.save(str(model_path))

# Save artifacts
artifacts = {
'cat_indices': cat_indices,
'standard_scaler': scalers['standard'],
'train_stats': train_stats,
'anomaly_threshold': threshold,

# Feature lists (important for inference)
'amount_features': AMOUNT_FEATURES,
'count_features': COUNT_FEATURES,
'temporal_features': TEMPORAL_FEATURES,
'binary_features': BINARY_FEATURES,
'cat_features': CATEGORICAL_FEATURES,
'num_features': num_features,
'total_features': total_features,
'group_cols': GROUP_COLS,

# Configuration snapshot
'encoder_config': ENCODER_CONFIG,
'decoder_config': DECODER_CONFIG,
'attention_config': ATTENTION_CONFIG,
'anomaly_config': ANOMALY_DETECTION_CONFIG
}

scalers_path = get_scalers_path()
save_artifacts(artifacts, str(MODELS_DIR), 'scalers.pkl')

print(f"\n✓ All artifacts saved:")
print(f"  Model: {model_path}")
print(f"  Scalers & Stats: {scalers_path}")

# COMMAND ----------

# VISUALIZATIONS
print("\n" + "="*80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("="*80)

# Plot training history
history_plot_path = get_results_path('training_history.png')
plot_training_history(history, str(history_plot_path))

# Plot error distribution
error_plot_path = get_results_path('error_distribution.png')
plot_reconstruction_errors(
combined_error,
threshold,
str(error_plot_path),
title='Reconstruction Error Distribution (Train + Val)'
)

# COMMAND ----------

# SUMMARY
print(f"\nModel Performance:")
print(f"  Final Train Loss: {history.history['loss'][-1]:.4f}")
print(f"  Final Val Loss: {history.history['val_loss'][-1]:.4f}")
print(f"  Loss Gap: {(history.history['val_loss'][-1] / history.history['loss'][-1]):.2f}x")
print(f"\nAnomaly Detection:")
print(f"  Threshold ({ANOMALY_DETECTION_CONFIG['threshold_method']}): {threshold:.4f}")
print(f"  Training Anomalies: {n_anomalies:,} ({anomaly_pct:.2f}%)")
print(f"\nSaved Files:")
print(f"  Model: {model_path.name}")
print(f"  Artifacts: {scalers_path.name}")
print(f"  Plots: {history_plot_path.name}, {error_plot_path.name}")

# COMMAND ----------

mlflow.set_registry_uri("databricks")

# COMMAND ----------

from src.inference_4 import *
def _log_model(
        model: AutoencoderModel,
        X_sample: List[np.ndarray],
        artifacts: Dict[str, Any],
        df_sample: pd.DataFrame,
        register_model: bool = True
    ):
    """
    Log the trained model to MLflow as pyfunc.
    
    Args:
        model: Trained AutoencoderModel
        X_sample: Sample input for signature inference
        artifacts: Dictionary of artifacts to log
        df_sample: Sample DataFrame for input signature
        register_model: Whether to register model in registry
    """
    # Create a temporary directory for artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # =========================================================
        # Save all artifacts needed for pyfunc wrapper
        # =========================================================
        
        # 1. Save Keras model
        keras_model_path = tmp_path / "BankAnomaly.keras"
        model.model.save(str(keras_model_path))
        
        # 2. Save all artifacts in a single pickle file
        artifacts_pkl_path = tmp_path / "artifacts.pkl"
        artifacts_dict = {
            'scaler_cat': artifacts['standard_scaler'],
            'scaler_num': artifacts['standard_scaler'],
            'train_stats': artifacts['train_stats'],
            'threshold': artifacts['anomaly_threshold'],
            'cat_indices': artifacts['cat_indices'],
            'config': {
                'categorical_features': artifacts['cat_features'],
                'numerical_features': artifacts['num_features'],
                'total_features': artifacts['total_features'],
                'binary_features': artifacts.get('binary_features', [])
            }
        }
        with open(artifacts_pkl_path, 'wb') as f:
            pickle.dump(artifacts_dict, f)
        
        # =========================================================
        # Create artifacts dict for pyfunc
        # =========================================================
        pyfunc_artifacts = {
            "model": str(keras_model_path),
            "artifacts": str(artifacts_pkl_path)
        }
        
        # =========================================================
        # Create conda environment
        # =========================================================
        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                f'python={sys.version_info.major}.{sys.version_info.minor}',
                'pip',
                {
                    'pip': [
                        f'mlflow=={mlflow.__version__}',
                        'tensorflow>=2.13.0',
                        'pandas>=1.5.0',
                        'numpy>=1.23.0',
                        'scikit-learn>=1.2.0',
                        'pyyaml>=6.0',
                        'cloudpickle>=2.2.0'
                    ]
                }
            ],
            'name': 'bank_anomaly_env'
        }
        
        # =========================================================
        # Create model signature
        # =========================================================
        # try:
        # Sample input: original data with all features
        input_sample = df_sample[REQUIRED_COLUMNS].head(5)
        
        # Create wrapper instance for prediction
        wrapper = BankAnomalyDetectorWrapper()
        
        # Manually set attributes
        wrapper.model = model.model
        wrapper.scaler_cat = artifacts['standard_scaler']
        wrapper.scaler_num = artifacts['standard_scaler']
        wrapper.train_stats = artifacts['train_stats']
        wrapper.threshold = artifacts['anomaly_threshold']
        wrapper.cat_indices = artifacts['cat_indices']
        wrapper.config = artifacts_dict['config']
        
        # Get output sample
        output_sample = wrapper.predict(None, input_sample)
        
        # Create signature
        signature = infer_signature(input_sample, output_sample)
        
        print("✓ Created model signature with input/output schema")
        print(f"  Input: {list(input_sample.columns)[:5]}... ({len(input_sample.columns)} features)")
        print(f"  Output: {list(output_sample.columns)}")
        # except Exception as e:
        #     print(f"⚠ Could not create signature: {e}")
        #     signature = None
        
        # =========================================================
        # Log pyfunc model
        # =========================================================
        # try:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=BankAnomalyDetectorWrapper(),
            artifacts=pyfunc_artifacts,
            conda_env=conda_env,
            signature=signature,
            code_paths=['src'],  # Include src code
            registered_model_name="bank_anomaly_detector" if register_model else None
        )
        print("✓ Model logged as pyfunc with inference wrapper")
        # =========================================================
        # Log additional artifacts
        # =========================================================
        
        # Save full artifacts dict (for backward compatibility)
        full_artifacts_path = tmp_path / "full_artifacts.pkl"
        with open(full_artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)
        mlflow.log_artifact(str(full_artifacts_path), "artifacts")
        
        # Log model summary as text
        summary_path = tmp_path / "model_summary.txt"
        with open(summary_path, 'w') as f:
            model.model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(str(summary_path), "model_info")
        
        # Log configuration summary
        config_summary_path = tmp_path / "model_config_summary.json"
        with open(config_summary_path, 'w') as f:
            json.dump(model.get_config_summary(), f, indent=2)
        mlflow.log_artifact(str(config_summary_path), "model_info")

# COMMAND ----------

with mlflow.start_run(run_name='Bk_anomaly') as run:
    _log_model(autoencoder_model, X_train_all, artifacts, df)
