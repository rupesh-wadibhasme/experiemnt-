"""
Standalone Transaction Anomaly Detection System
================================================
A simplified end-to-end pipeline for training and inference of an autoencoder-based 
anomaly detection model. Designed for educational purposes and junior team training.

Features:
---------
1. Sample transaction data generation
2. Simple preprocessing and feature engineering
3. Multi-input autoencoder model training (50 epochs)
4. Model saving and loading
5. Inference with reconstruction error calculation
6. Anomaly detection with configurable threshold (default: 50)

Usage:
------
python standalone_anomaly_detection.py
"""

import pandas as pd
import numpy as np
import keras
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, GaussianNoise
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================================

def identify_column_types(df):
    """Separates categorical and numeric columns."""
    categorical_cols = []
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in (object, str):
            categorical_cols.append(col)
        elif df[col].dtype in (float, int):
            numeric_cols.append(col)
    return categorical_cols, numeric_cols


def encode_categorical(df, encoder=None):
    """Applies one-hot encoding to categorical features."""
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df)
    encoded_data = encoder.transform(df)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
    return encoded_df, encoder


def normalize_numeric(df, scaler=None):
    """Normalizes numeric features using Min-Max scaling."""
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)
    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
    return scaled_df, scaler


def calculate_transaction_value(row):
    """Calculates transaction value based on base currency."""
    if row['BaseCurrency'] == row['BuyCurrency']:
        return np.abs(row['BuyValue'])
    elif row['BaseCurrency'] == row['SellCurrency']:
        return np.abs(row['SellValue'])
    return 0.0


def create_train_test_split(df):
    """Creates train/test split indices."""
    train_idx, test_idx = train_test_split(
        df.index, 
        test_size=0.2, 
        random_state=42
    )
    return {'train': train_idx, 'test': test_idx}


def prepare_model_inputs(features, numeric_data, categorical_cols, split_indices):
    """Prepares input data for multi-input model."""
    train_idx = split_indices['train']
    test_idx = split_indices['test']
    
    train_inputs = []
    test_inputs = []
    input_dimensions = []
    
    # Prepare categorical inputs (one input per category)
    for category in categorical_cols:
        cat_cols = [x for x in features.columns if x.startswith(category)]
        cat_df = features[cat_cols]
        
        train_inputs.append(cat_df.loc[train_idx].values)
        test_inputs.append(cat_df.loc[test_idx].values)
        input_dimensions.append(len(cat_cols))
    
    # Prepare numeric input (all numeric features together)
    train_inputs.append(numeric_data.loc[train_idx].values)
    test_inputs.append(numeric_data.loc[test_idx].values)
    input_dimensions.append(numeric_data.shape[1])
    
    return train_inputs, test_inputs, input_dimensions


# ============================================================================
# SECTION 2: MODEL ARCHITECTURE
# ============================================================================

def build_autoencoder_model(input_dimensions, hidden_layers, activation, noise_level, 
                            dropout_rate=0, regularizer=None):
    """Builds multi-input autoencoder model."""
    # Create input layers for each feature group
    input_layers = [Input(shape=(dim,)) for dim in input_dimensions]
    
    # Add Gaussian noise to inputs for robustness
    noisy_inputs = [GaussianNoise(stddev=noise_level)(layer) for layer in input_layers]
    
    # First hidden layer processes each input separately
    hidden = [Dense(hidden_layers[0], activation="relu", kernel_regularizer=regularizer)(noise) 
              for noise in noisy_inputs]
    
    # Concatenate all processed inputs
    hidden = Concatenate()(hidden)
    
    # Additional hidden layers
    for neurons in hidden_layers[1:]:
        hidden = Dense(neurons, kernel_regularizer=None)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(dropout_rate)(hidden)
    
    # Output layers: reconstruct each input
    categorical_outputs = [
        Dense(dim, activation=activation, kernel_regularizer=regularizer, 
              name=f'categorical_output_{i}')(hidden) 
        for i, dim in enumerate(input_dimensions[:-1])
    ]
    
    numeric_output = [
        Dense(input_dimensions[-1], kernel_regularizer=regularizer, 
              name='numeric_output')(hidden)
    ]
    
    all_outputs = categorical_outputs + numeric_output
    
    model = keras.Model(inputs=input_layers, outputs=all_outputs)
    return model


def train_autoencoder(train_inputs, test_inputs, model, optimizer, loss_functions, 
                     loss_weights, epochs, batch_size, steps_per_epoch):
    """Trains the autoencoder model."""
    model.compile(
        loss=loss_functions,
        loss_weights=loss_weights,
        optimizer=optimizer,
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.03,
        patience=5,
        restore_best_weights=True
    )
    
    print("\n" + "="*70)
    print("TRAINING AUTOENCODER MODEL")
    print("="*70)
    
    history = model.fit(
        train_inputs, 
        train_inputs,  # Autoencoder: input = output
        epochs=epochs,
        validation_data=(test_inputs, test_inputs),
        verbose=1,
        callbacks=[early_stopping],
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )
    
    print("\n✓ Training completed!")
    return model, history


# ============================================================================
# SECTION 3: SAMPLE DATA GENERATION
# ============================================================================

def generate_sample_transactions(n_samples=1000):
    """Generates synthetic transaction data for testing."""
    print("\n" + "="*70)
    print("GENERATING SAMPLE TRANSACTION DATA")
    print("="*70)
    
    np.random.seed(42)
    
    # Define possible values
    transaction_types = ['Type_A', 'Type_B', 'Type_C']
    departments = ['Dept_001', 'Dept_002', 'Dept_003']
    partners = ['Partner_W', 'Partner_X', 'Partner_Y', 'Partner_Z']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        transaction_date = start_date + timedelta(days=np.random.randint(0, 365))
        days_to_settlement = np.random.choice([1, 7, 30, 90, 180, 360])
        settlement_date = transaction_date + timedelta(days=days_to_settlement)
        
        buy_currency = np.random.choice(currencies)
        sell_currency = np.random.choice([c for c in currencies if c != buy_currency])
        base_currency = np.random.choice([buy_currency, sell_currency])
        
        buy_value = np.random.uniform(10000, 1000000)
        exchange_rate = np.random.uniform(0.5, 2.0)
        sell_value = buy_value * exchange_rate
        rate_adjustment = np.random.uniform(-0.01, 0.01)
        
        # Determine if weekend
        is_weekend = 1 if transaction_date.isoweekday() >= 6 else 0
        
        data.append({
            'TransactionType': np.random.choice(transaction_types),
            'Department': np.random.choice(departments),
            'Partner': np.random.choice(partners),
            'TransactionDate': transaction_date,
            'SettlementDate': settlement_date,
            'BaseCurrency': base_currency,
            'BuyValue': buy_value,
            'SellValue': sell_value,
            'BuyCurrency': buy_currency,
            'SellCurrency': sell_currency,
            'ExchangeRate': exchange_rate,
            'RateAdjustment': rate_adjustment,
            'IsWeekend': is_weekend,
            'DaysToSettlement': days_to_settlement
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} sample transactions")
    print(f"  Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
    print(f"  Unique partners: {df['Partner'].nunique()}")
    print(f"  Unique currencies: {len(set(df['BuyCurrency'].unique()) | set(df['SellCurrency'].unique()))}")
    
    return df


# ============================================================================
# SECTION 4: TRAINING PIPELINE
# ============================================================================

def training_pipeline(data, models_path, epochs=50, model_name='transaction_model'):
    """Complete training pipeline with simplified preprocessing."""
    print("\n" + "="*70)
    print("STARTING TRAINING PIPELINE")
    print("="*70)
    
    # Create models directory
    os.makedirs(models_path, exist_ok=True)
    
    # Select required columns
    required_cols = ['TransactionType', 'Department', 'Partner', 'TransactionDate',
                    'SettlementDate', 'BaseCurrency', 'BuyValue', 'SellValue', 
                    'BuyCurrency', 'SellCurrency', 'ExchangeRate', 'RateAdjustment']
    
    # Preprocessing
    print("\n→ Preprocessing data...")
    data = data[required_cols].copy()
    
    # Remove zero-value transactions
    data = data[(data.BuyValue != 0.0) & (data.SellValue != 0.0)]
    data = data.drop_duplicates(keep='last')
    
    # Simple feature engineering
    print("→ Engineering features...")
    
    # Calculate days to settlement
    data['DaysToSettlement'] = (data.SettlementDate - data.TransactionDate).dt.days
    
    # Check if transaction is on weekend
    data['IsWeekend'] = data.TransactionDate.apply(lambda x: 1 if x.isoweekday() >= 6 else 0)
    
    # Convert Department to string (treat as categorical)
    data['Department'] = data['Department'].astype(str)
    
    # Calculate transaction value based on base currency
    data['TransactionValue'] = data.apply(calculate_transaction_value, axis=1)
    
    # Drop date columns (already extracted features)
    data = data.drop(['TransactionDate', 'SettlementDate'], axis=1)
    
    # Fill any missing values
    data.fillna(0, inplace=True)
    
    # Identify column types
    print("→ Identifying feature types...")
    categorical_cols, numeric_cols = identify_column_types(data)
    
    print(f"  Categorical features: {categorical_cols}")
    print(f"  Numeric features: {numeric_cols}")
    
    # Separate and encode categorical features
    print("→ Encoding categorical features...")
    categorical_data, encoder = encode_categorical(data[categorical_cols])
    
    # Normalize numeric features
    print("→ Normalizing numeric features...")
    numeric_data, scaler = normalize_numeric(data[numeric_cols])
    
    # Combine all features
    all_features = pd.concat([categorical_data, numeric_data], axis=1)
    
    # Save preprocessing artifacts
    print("→ Saving preprocessing artifacts...")
    artifacts = {
        model_name: {
            'scaler': scaler,
            'encoder': encoder,
            'categorical_cols': categorical_cols,
            'numeric_cols': numeric_cols
        }
    }
    
    with open(os.path.join(models_path, "preprocessing_artifacts.pkl"), 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Prepare inputs for model
    print("→ Preparing model inputs...")
    split_indices = create_train_test_split(data)
    train_inputs, test_inputs, input_dims = prepare_model_inputs(
        all_features, numeric_data, categorical_cols, split_indices
    )
    
    # Calculate loss weights (equal weight per feature)
    total_features = len(input_dims[:-1]) + input_dims[-1]
    weight_per_feature = 1.0 / total_features
    loss_weights = [weight_per_feature] * len(categorical_cols)
    loss_weights.append(input_dims[-1] * weight_per_feature)
    
    # Build model
    print("→ Building autoencoder architecture...")
    hidden_layers = [64, 32]
    model = build_autoencoder_model(
        input_dims,
        hidden_layers,
        activation='softmax',
        noise_level=0.1,
        dropout_rate=0.2,
        regularizer=keras.regularizers.l2(0.01)
    )
    
    print(f"\n  Model architecture:")
    print(f"  - Input dimensions: {input_dims}")
    print(f"  - Hidden layers: {hidden_layers}")
    print(f"  - Total parameters: {model.count_params():,}")
    
    # Define loss functions
    loss_funcs = ['categorical_crossentropy'] * len(categorical_cols) + ['mse']
    
    # Train model
    model, history = train_autoencoder(
        train_inputs,
        test_inputs,
        model,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss_functions=loss_funcs,
        loss_weights=loss_weights,
        epochs=epochs,
        batch_size=32,
        steps_per_epoch=100
    )
    
    # Save model
    model_path = os.path.join(models_path, f"{model_name}.keras")
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return model, artifacts, categorical_cols, numeric_cols


# ============================================================================
# SECTION 5: INFERENCE AND ANOMALY DETECTION
# ============================================================================

def load_trained_model(models_path, model_name='transaction_model'):
    """Loads trained model and preprocessing artifacts."""
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    model_path = os.path.join(models_path, f"{model_name}.keras")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    
    with open(os.path.join(models_path, "preprocessing_artifacts.pkl"), 'rb') as f:
        artifacts = pickle.load(f)
    print("✓ Preprocessing artifacts loaded")
    
    return model, artifacts[model_name]


def preprocess_for_inference(data, artifacts):
    """Preprocesses data for inference using saved artifacts."""
    required_cols = ['TransactionType', 'Department', 'Partner', 'TransactionDate',
                    'SettlementDate', 'BaseCurrency', 'BuyValue', 'SellValue', 
                    'BuyCurrency', 'SellCurrency', 'ExchangeRate', 'RateAdjustment']
    
    data = data[required_cols].copy()
    
    # Feature engineering (same as training)
    data['DaysToSettlement'] = (data.SettlementDate - data.TransactionDate).dt.days
    data['IsWeekend'] = data.TransactionDate.apply(lambda x: 1 if x.isoweekday() >= 6 else 0)
    data['Department'] = data['Department'].astype(str)
    data['TransactionValue'] = data.apply(calculate_transaction_value, axis=1)
    
    # Drop date columns
    data = data.drop(['TransactionDate', 'SettlementDate'], axis=1)
    data.fillna(0, inplace=True)
    
    # Get categorical and numeric columns from artifacts
    categorical_cols = artifacts['categorical_cols']
    numeric_cols = artifacts['numeric_cols']
    
    # Encode and normalize using saved transformers
    categorical_data, _ = encode_categorical(data[categorical_cols], artifacts['encoder'])
    numeric_data, _ = normalize_numeric(data[numeric_cols], artifacts['scaler'])
    
    # Prepare inputs for model
    input_list = []
    for category in categorical_cols:
        cat_cols = [x for x in categorical_data.columns if x.startswith(category)]
        input_list.append(categorical_data[cat_cols].values)
    input_list.append(numeric_data.values)
    
    all_features = pd.concat([categorical_data, numeric_data], axis=1)
    
    return input_list, all_features


def compute_reconstruction_error(model, input_list, features):
    """Computes reconstruction error for anomaly detection."""
    # Get model predictions
    predictions = model.predict(input_list, verbose=0)
    
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Reconstruct features from predictions
    reconstructed_df = pd.DataFrame()
    
    for i, pred in enumerate(predictions):
        if i < len(predictions) - 1:  # Categorical outputs
            # Convert probabilities to one-hot (argmax)
            reconstructed = np.array([np.where(row == max(row), 1.0, 0.0) for row in pred])
        else:  # Numeric output
            reconstructed = pred
        
        reconstructed_df = pd.concat([reconstructed_df, pd.DataFrame(reconstructed)], axis=1)
    
    reconstructed_df.columns = features.columns
    
    # Calculate Mean Squared Error per transaction
    reconstruction_errors = np.mean((features.values - reconstructed_df.values) ** 2, axis=1)
    
    return reconstruction_errors, reconstructed_df


def detect_anomalies(data, models_path, threshold=50, model_name='transaction_model'):
    """Performs inference and detects anomalies based on reconstruction error."""
    print("\n" + "="*70)
    print("ANOMALY DETECTION")
    print("="*70)
    
    # Load model and artifacts
    model, artifacts = load_trained_model(models_path, model_name)
    
    # Preprocess data
    print("\n→ Preprocessing inference data...")
    input_list, features = preprocess_for_inference(data, artifacts)
    
    # Calculate reconstruction errors
    print("→ Computing reconstruction errors...")
    reconstruction_errors, reconstructed_features = compute_reconstruction_error(
        model, input_list, features
    )
    
    # Detect anomalies based on threshold
    print(f"→ Detecting anomalies (threshold: {threshold})...")
    is_anomaly = reconstruction_errors > threshold
    
    # Create results dataframe
    results = data.copy()
    results['reconstruction_error'] = reconstruction_errors
    results['is_anomaly'] = is_anomaly
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    print(f"Total transactions analyzed: {len(results)}")
    print(f"Anomalies detected: {is_anomaly.sum()} ({100*is_anomaly.sum()/len(results):.2f}%)")
    print(f"Normal transactions: {(~is_anomaly).sum()} ({100*(~is_anomaly).sum()/len(results):.2f}%)")
    
    print(f"\nReconstruction error statistics:")
    print(f"  Mean: {reconstruction_errors.mean():.4f}")
    print(f"  Median: {np.median(reconstruction_errors):.4f}")
    print(f"  Min: {reconstruction_errors.min():.4f}")
    print(f"  Max: {reconstruction_errors.max():.4f}")
    print(f"  Std Dev: {reconstruction_errors.std():.4f}")
    
    if is_anomaly.sum() > 0:
        print(f"\nTop 5 anomalies by reconstruction error:")
        top_anomalies = results[results['is_anomaly']].nlargest(5, 'reconstruction_error')
        for idx, row in top_anomalies.iterrows():
            print(f"  - Transaction {idx}: Error = {row['reconstruction_error']:.4f}")
            print(f"    Partner: {row['Partner']} | {row['BuyCurrency']}/{row['SellCurrency']} | "
                  f"Value: {row['BuyValue']:.2f}")
    
    return results


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function - runs complete pipeline."""
    print("\n" + "="*70)
    print("TRANSACTION ANOMALY DETECTION - COMPLETE PIPELINE")
    print("="*70)
    
    # Configuration
    models_path = "./anomaly_detection_models"
    model_name = "transaction_model"
    training_epochs = 50
    anomaly_threshold = 50
    
    # Step 1: Generate training data
    print("\n[STEP 1/5] Generating training data...")
    train_data = generate_sample_transactions(n_samples=1000)
    
    # Step 2: Train model
    print("\n[STEP 2/5] Training autoencoder model...")
    model, artifacts, categorical_cols, numeric_cols = training_pipeline(
        train_data, 
        models_path, 
        epochs=training_epochs,
        model_name=model_name
    )
    
    # Step 3: Generate test data
    print("\n[STEP 3/5] Generating test data for inference...")
    test_data = generate_sample_transactions(n_samples=100)
    print(f"✓ Generated {len(test_data)} test transactions")
    
    # Step 4: Run anomaly detection
    print("\n[STEP 4/5] Running anomaly detection...")
    results = detect_anomalies(
        test_data,
        models_path,
        threshold=anomaly_threshold,
        model_name=model_name
    )
    
    # Step 5: Save results
    print("\n[STEP 5/5] Saving results...")
    results_path = os.path.join(models_path, "anomaly_detection_results.csv")
    results.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated artifacts in: {models_path}/")
    print(f"  - {model_name}.keras (trained model)")
    print(f"  - preprocessing_artifacts.pkl (scalers & encoders)")
    print(f"  - anomaly_detection_results.csv (detection results)")
    print("\nYou can now use this model to detect anomalies in new transaction data!")
    
    return results


if __name__ == "__main__":
    results = main()
