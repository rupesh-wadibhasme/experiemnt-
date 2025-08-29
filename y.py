# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install openpyxl keras tensorflow iterative-stratification flask mlflow==3.0.1 azure-identity==1.19.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.tensorflow
mlflow.set_registry_uri("databricks")

# COMMAND ----------

import os
import keras
import pickle
import datetime
import numpy as np
import pandas as pd
from keras.src.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, GaussianNoise
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import random
from datetime import datetime
import mlflow
import mlflow.deployments
import time
from time import sleep
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving
from mlflow.deployments import get_deploy_client
import yaml
import tensorflow as tf
from functools import reduce
from glob import glob
random.seed(42)

# COMMAND ----------

import pre_process

# COMMAND ----------

def get_model_to_train(input_shapes, neuron_set,
                       activation, noise, dropout=0, regularizer=None,):
  input_layers = [Input(shape=(shape,)) for shape in input_shapes]
  noise_inputs = [GaussianNoise(stddev=noise)(layer) for layer in input_layers]
  hidden_output = [Dense(neuron_set[0], activation="relu", kernel_regularizer=regularizer)(noise) for noise in noise_inputs]
  hidden_output = Concatenate()(hidden_output)

  for neuron in neuron_set[1:]:
    hidden_output = Dense(neuron, kernel_regularizer=None)(hidden_output)
    hidden_output = BatchNormalization()(hidden_output)
    hidden_output = keras.layers.Dropout(dropout)(hidden_output)

  output_cat = [Dense(shape, activation=activation, kernel_regularizer=regularizer,name=f'catg_out-{i}')(hidden_output) for i,shape in enumerate(input_shapes[0:-1])]
  output_num = [Dense(input_shapes[-1], kernel_regularizer=regularizer,name='numeric_out')(hidden_output)]
  outputs = output_cat+output_num
  model = keras.Model(inputs=input_layers, outputs=outputs)
  return model

  

def train_model(input_data_train, input_data_test, model,
                optimizer,
                loss_funcs,
                loss_weights,
                epochs,
                batch_size,
                steps_per_epoch,
                test_num,
                client,
                training=True
                ):
  model.compile(loss=loss_funcs, 
          loss_weights=loss_weights,
          optimizer=optimizer,
          )
  csvlogger_path = f'training_logs_{datetime.today()}_{client}_{test_num}'
  csv_logger = keras.callbacks.CSVLogger(os.path.join(models_path, csvlogger_path), separator=",", append=False)

  history = model.fit(input_data_train, input_data_train, epochs=epochs,
                  validation_data = (input_data_test, input_data_test),verbose=1,
                  callbacks=[csv_logger],
                  batch_size = batch_size,
                  steps_per_epoch = steps_per_epoch
                  )
  learning_curve = pd.DataFrame(history.history)
  def plot_and_save_logs(test_num):
          total_loss_size = len(learning_curve.columns)
          loss_tuples = [(learning_curve.columns[[idx,idx+8]]) for idx in range(int(len(learning_curve.columns.to_list())/2))]
          fig = learning_curve.plot(subplots = loss_tuples, figsize = (8,30),)[0].get_figure()
          fig_name = os.path.join(models_path, f'logs_plot.png')
          fig.savefig(fig_name, dpi=300)
          return fig_name
  fig_name = plot_and_save_logs(test_num=test_num)
  return model, fig_name


# COMMAND ----------

def training(data, models_path, epochs, neuron_set, activation, noise, dropout, optimizer, loss_funcs, steps_per_epoch, regularizer, batch_size, test_num=1,client_name='rcc'):
  # Read the columns that are decided already.
  col_list = ['FX_Instrument_Type', 'BUnit', 'Cpty','DealDate',
      'MaturityDate', 'PrimaryCurr','BuyAmount',
      'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']
  categories_in_numerics = ['BUnit']
  groupby_columns_names = ['BUnit', 'Cpty', 'PrimaryCurr']
  group_trantype = ['FX_Instrument_Type']

  def get_column_types(df):
    categorical_columns = []
    numeric_columns = []
    for feature in df.columns:
        if df[feature].dtype == object:
          categorical_columns.append(feature)
        elif df[feature].dtype in (float, int) :
          numeric_columns.append(feature)
    return categorical_columns, numeric_columns

  def one_hot(df, encoder=None):
    if encoder is None:
      encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
      encoder.fit(df)
    encoded_data = encoder.transform(df)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
    return encoded_df, encoder

  def scale(df, scaler=None):
    if scaler is None:
      scaler = MinMaxScaler()
      scaler.fit(df)
    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
    return scaled_df, scaler
  
  def remove_list(original_list, remove_list):
    for i in remove_list:
      original_list.remove(i)
    return original_list
  
  def face_value(df):
    df["FaceValue"] = np.nan
    if df.PrimaryCurr == df.BuyCurr:
        df["FaceValue"]=np.abs(df.BuyAmount)
    elif df.PrimaryCurr == df.SellCurr:
        df["FaceValue"]=np.abs(df.SellAmount)
    return df
  
  def minmax_scale_group(group, column):
    scaler = MinMaxScaler()
    group[[column]] = scaler.fit_transform(group[[column]])
    mean = float(group[column].mean())
    std_dev = float(group[column].std())
    return group, scaler, mean, std_dev

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


  def group_points(df, groupby_columns, column, percentile=False):
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
          # print(name, percent)
        if percentile:
          group, lower, upper, counts = clip_percentile(group, column, upper_percentile=0.99)
          percentile_lt[name] = {'lb':lower, 'ub':upper}
        
        total_counts+=counts[0]
        lower_counts+=counts[1]
        upper_counts+=counts[2]
        scaled_group, scaler, mean, sd = minmax_scale_group(group, column)
        # grouped_scalers[name] = {'scaler':scaler, 'mean':mean, 'sd':sd}
        grouped_scalers[name] = scaler
        grouped_scaled_dfs.append(scaled_group)
    grouped_df = pd.concat(grouped_scaled_dfs)
    return grouped_df, grouped_scalers, (total_counts, lower_counts, upper_counts), (iqr_bds, zscore_bds, percentile_lt, mod_zscore_lt)
  
  def split_data(df):
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    for idx, (train_index, test_index) in enumerate(kf.split(df, df), start=1):
        df.loc[test_index, 'fold'] = idx
    return df
  
  def prepare_inputs(features, num_data, categorical_columns, df_split):
    input_data_train = []
    input_data_test = []
    input_shapes = []
    train_index = df_split[df_split['fold']!=5].index
    test_index = df_split[df_split['fold']==5].index
    num_cols = num_data.columns.tolist()
    for category in categorical_columns:
      category_cols = [x for x in features.columns if x.startswith(category)]
      input_shapes.append(len(category_cols))
      category_df = features[category_cols]

      train_data, test_data = category_df.loc[train_index, :], category_df.loc[test_index, :]
      assert train_data.shape[1]==test_data.shape[1]
      assert train_data.columns.tolist()==test_data.columns.tolist()
      input_data_train.append(train_data)
      input_data_test.append(test_data)
    input_shapes.append(len(num_cols))
    input_data_train.append(features[num_cols].loc[train_index, :])
    input_data_test.append(features[num_cols].loc[test_index, :])
    return input_data_train, input_data_test, input_shapes

  data.rename(columns={'ActivationDate':'DealDate'}, inplace=True)
  data.rename(columns={'Instrument':'FX_Instrument_Type'}, inplace=True)
  data = data[col_list].copy()
  #Remove unwanted rows
  data = data[(data.BuyAmount != 0.0) & (data.SellAmount != 0.0)]

  #Remove the contract Number, rownum from features to identify the duplicate rows
  cols = data.columns.to_list()
  data = data.drop_duplicates(keep='last',subset=cols)
  data['Is_weekend_date'] = data.DealDate.apply(lambda x: x.date().isoweekday())
  #Convert weekdays to '0' and weekend to '1'
  data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x<6 else 1)
  data['TDays'] = (data.MaturityDate - data.DealDate).dt.days
  #Convert BUnit, TranCode, AuthorisedStatus, DealerID into Categorical.
  data[categories_in_numerics] = data[categories_in_numerics].astype(str)
  #Identify categorical Columns and Numeric columns
  categorical_columns, numeric_columns = get_column_types(data)
  data = pd.concat((data[categorical_columns].astype('category'), data[numeric_columns].astype(float)), axis=1)
  #Facevalue column creation based on Primary Currency
  data = data.apply(face_value, axis=1)
  cpty_groups = {}
  for cp in data['Cpty'].unique().tolist():
      buy_unique = data[data['Cpty']==cp]['BuyCurr'].unique().tolist()
      sell_unique = data[data['Cpty']==cp]['SellCurr'].unique().tolist()
      cpty_groups[cp] = {'buy': buy_unique, 'sell': sell_unique}

  data.fillna(0, inplace=True)

  grouped_df, fv_scalers, total_counts, limits = group_points(data, groupby_columns_names, 'FaceValue',percentile=True)
  grouped_df, tdays_scalers, total_counts, limits = group_points(grouped_df, group_trantype, 'TDays', percentile=True)#zscore 
  numeric_columns = remove_list(numeric_columns, ['TDays'])
  grouped_df = grouped_df.replace([np.inf, -np.inf], np.nan)
  
  cat_data, ohe = one_hot(grouped_df[categorical_columns])
  num_data, mms = scale(grouped_df[numeric_columns])
  num_data['FaceValue'] = grouped_df['FaceValue'].values
  num_data['TDays'] = grouped_df['TDays'].values
  features = pd.concat([cat_data, num_data,], axis=1)
  features = features.replace([np.inf, -np.inf], np.nan)
  pickle.dump({client_name: {'mms':mms, 'ohe':ohe, 'grouped_scalers': fv_scalers, 'cpty_group': cpty_groups, 'tdays_scalers': tdays_scalers}}, open(os.path.join(models_path, "all_scales.pkl"), 'wb'))

  df_split = split_data(grouped_df[['BUnit', 'Cpty']].reset_index(drop=True))
  input_data_train, input_data_test, input_shapes = prepare_inputs(features, num_data, categorical_columns, df_split)

  total_features = len(input_shapes[:-1]) + input_shapes[-1]
  loss_weight_per_feature = (1/total_features)
  loss_weights= [loss_weight_per_feature] * len(categorical_columns)
  loss_weights.append(input_shapes[-1]*loss_weight_per_feature)

  model = get_model_to_train(input_shapes, neuron_set, 
          activation,
          noise,
          dropout=dropout,
          regularizer=regularizer,
          )

  model, fig_name = train_model(input_data_train, input_data_test, model,
                  optimizer,
                  loss_funcs,
                  loss_weights,
                  epochs,
                  batch_size,
                  steps_per_epoch,
                  test_num,
                  client_name,
                  )
  return model, fig_name

# COMMAND ----------

def load_config(client_name, config_file='config.yml'):
    """
    Loads configuration from a YAML file, applying client-specific overrides.
    """
    with open(config_file, 'r') as file:
        full_config = yaml.safe_load(file)

    default_config = full_config.get('default_config', {})
    client_configs = full_config.get('clients', {})

    # Start with default config
    config = default_config.copy()

    # Apply client-specific overrides
    if client_name in client_configs:
        client_specific = client_configs[client_name]
        for key, value in client_specific.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                # Merge nested dictionaries (e.g., for optimizer, regularizer)
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    else:
        print(f"Warning: Client '{client_name}' not found in config. Using default settings.")

    # Post-process to create Keras objects
    processed_config = {}
    for key, value in config.items():
        if key == 'regularizer' and isinstance(value, dict):
            if value.get('type') == 'l2':
                processed_config['regularizer'] = keras.regularizers.l2(value.get('value', 0.01))
            elif value.get('type') == 'l1':
                processed_config['regularizer'] = keras.regularizers.l1(value.get('value', 0.01))
            # Add other regularizer types as needed
        elif key == 'optimizer' and isinstance(value, dict):
            if value.get('type') == 'Adam':
                processed_config['optimizer'] = keras.optimizers.Adam(learning_rate=value.get('learning_rate', 1e-3))
            # Add other optimizer types as needed
        elif key == 'loss_funcs' and isinstance(value, list):
            processed_loss_funcs = []
            for loss_item in value:
                if isinstance(loss_item, dict) and loss_item.get('type') == 'CategoricalCrossentropy':
                    processed_loss_funcs.append(keras.losses.CategoricalCrossentropy())
                elif isinstance(loss_item, dict) and loss_item.get('type') == 'mse':
                    processed_loss_funcs.append('mse') # Keras accepts 'mse' string directly
                # Add other loss types as needed
            processed_config['loss_funcs'] = processed_loss_funcs
        else:
            processed_config[key] = value
            
    # Add base_path which is runtime dependent
    processed_config['base_path'] = os.getcwd()

    return processed_config

# COMMAND ----------

def list_and_read_parquet_files(folder_path):
    """
    List and read all parquet files in a given folder.
    """
    all_files = glob(os.path.join(folder_path, '*.parquet'))
    if not all_files:
        return []
    
    # Read and concatenate all parquet files
    dfs = [pd.read_parquet(file) for file in all_files]
    return dfs

# COMMAND ----------

conda_env = {
    'channels': ['conda-forge'],
    'dependencies': [
        'python=3.11.10', # Match your Databricks Runtime Python version
        'pip',
        {
            'pip': [
                'mlflow==3.0.1',
                'tensorflow',
                'keras', # Often installed with tensorflow, but good to be explicit
                'scikit-learn',
                'pandas',
                'pydantic',
                'flask',
                'azure-identity==1.19.0',
                'cloudpickle==2.2.1' # Ensure compatible version
            ]
        }
    ],
    'name': 'fx_anomaly_prediction_env'
}

# Define a single input example that matches ValidateParams for signature inference
input_example = [{
    "Client": "rcc",
    "FX_Instrument_Type": "Spot",
    "BUnit": "2",
    "Cpty": "RBST",
    "PrimaryCurr": "EUR",
    "BuyCurr": "EUR",
    "SellCurr": "USD",
    "BuyAmount": 1000.0,
    "SellAmount": 800.0,
    "SpotRate": 1.25,
    "ForwardPoints": 0.0,
    "DealDate": "2007-05-29",
    "MaturityDate": "2007-06-4",
    "UniqueId": "FX123456789"
}]

# COMMAND ----------

def prepare_model_registry(model, client, modelpath):
    artifact_path = f'model_{client}'
    model_name = f'FX_anomaly_prediction-{client}.keras'
    model.save(os.path.join(modelpath, model_name),overwrite=True)
    # registered_model_name = f"FX_anomaly_prediction-{client}"
    keras_model_local_path = os.path.join(modelpath, model_name)
    scalers_local_path = os.path.join(modelpath, "all_scales.pkl")
    model_params_for_endpoint = {"allowed_client": client}
    client_registered_model_full_name = f"{REGISTERED_CLIENT_MODEL_NAME_BASE}_{client}"
    get_or_create_registered_model(client_registered_model_full_name)
    with mlflow.start_run(run_name=f"FX Anomaly Prediction Training{client}") as run:
        artifacts = {'keras_autoencoder_path': keras_model_local_path, 'scalers_path': scalers_local_path}
        mlflow.pyfunc.log_model(
            python_model=pre_process.KerasAnomalyDetectorPyfunc(),
            name=artifact_path,
            conda_env=conda_env,
            artifacts=artifacts,
            code_paths=["pre_process.py"],
            registered_model_name=client_registered_model_full_name,
            model_config=model_params_for_endpoint,
        )

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"Model logged to: runs:/{run_id}/{artifact_path}")
        print(f"Model registered as: {client_registered_model_full_name}")
        return client_registered_model_full_name, run_id


# COMMAND ----------

def get_or_create_registered_model(model_name):
    """Checks if a registered model exists, otherwise creates it."""
    try:
        return client_tracking.get_registered_model(name=model_name)
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e): # Specific error for not found
            return client_tracking.create_registered_model(name=model_name)
        else:
            raise

# COMMAND ----------

REGISTERED_CLIENT_MODEL_NAME_BASE = "fx_anomaly_prediction_logger_test"
client_tracking = mlflow.tracking.MlflowClient()

# COMMAND ----------

try:
    registered_models_info = {}
    base_volume_path = "/Volumes/aimluat/treasury_forecasting" 
    # Iterate through each sub-folder in the base volume path
    for root, dirs, files in os.walk(base_volume_path):
        for client_folder in dirs:
            client_path = os.path.join(root, client_folder)
            # Check if any file in the folder name contains 'AI_FX'
            parquet_files = [f for f in os.listdir(client_path) if f.endswith('.parquet')]
            if any('AI_FX' in f for f in parquet_files):
                print(f"\nProcessing client folder: {client_folder}")
                
                # Use the folder name as the client identifier
                client = client_folder.split('_')[-1].upper()
                client_config = load_config(client)
                
                # Unpack the config
                base_path, regularizer, neuron_set, activation, noise, dropout, optimizer, loss_funcs, epochs, batch_size, steps_per_epoch = (
                    client_config['base_path'], client_config['regularizer'], client_config['neuron_set'],
                    client_config['activation'], client_config['noise'], client_config['dropout'],
                    client_config['optimizer'], client_config['loss_funcs'], client_config['epochs'],
                    client_config['batch_size'], client_config['steps_per_epoch']
                )

                # Read data from the folder
                data_frames = list_and_read_parquet_files(client_path)
                
                if data_frames:
                    models_path = os.path.join(base_path, client)
                    os.makedirs(models_path, exist_ok=True)
                    
                    # Concatenate all dataframes
                    combined_data = pd.concat(data_frames, ignore_index=True)
                    
                    # Train the model
                    model, fig_name = training(
                        combined_data, models_path, epochs, neuron_set, activation, noise,
                        dropout, optimizer, loss_funcs, steps_per_epoch, regularizer,
                        batch_size, client_name=client
                    )
                    registerd_model_name , run_id = prepare_model_registry(model, client, models_path)
                    if registerd_model_name:
                        registered_models_info[client] = {"model_name": registerd_model_name, "run_id": run_id}
except Exception as e:
    print(f"Failed to train and register model: {e}")
        

# COMMAND ----------

registered_models_info

# COMMAND ----------

DATABRICKS_HOST="https://adb-8520998198412329.9.azuredatabricks.net/"
for container, info in registered_models_info.items():
    try:
        model_name = info['model_name']
        run_id = info['run_id']

        client = mlflow.deployments.get_deploy_client("databricks")
        endpoint_name = model_name.lower()
        model_versions = client_tracking.search_model_versions(f"name='{model_name}'")
        # Sort the versions to find the latest
        latest_version = sorted(model_versions, key=lambda mv: int(mv.version), reverse=True)
        if latest_version:
            model_version_to_serve = latest_version[0].version
        else:
            raise ValueError(f"No version found for model '{model_name}'")
        print(f'Latest version used for deployment : {model_version_to_serve}')

        try:
            # Check if endpoint already exists
            endpoint_info = client.get_endpoint(endpoint_name)
            print(f"Endpoint '{endpoint_name}' already exists. Updating it.")
            # Update existing endpoint to serve the new version
            client.update_endpoint(
                endpoint=endpoint_name,
                config={
                    "served_models": [
                        {
                            "model_name": model_name,
                            "model_version": model_version_to_serve,
                            "workload_size": "Small", # e.g., "Small", "Medium", "Large"
                            "scale_to_zero_enabled": False,
                        }
                    ]
                }
            )
            print(f"Endpoint '{endpoint_name}' updated to serve version {model_version_to_serve}.")

        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e): # Or check specific error for not found
                print(f"Endpoint '{endpoint_name}' does not exist. Creating it.")
                # Create a new endpoint
                client.create_endpoint(
                    name=endpoint_name,
                    config={
                        "served_models": [
                            {
                                "model_name": model_name,
                                "model_version": model_version_to_serve,
                                "workload_size": "Small",
                                "scale_to_zero_enabled": False,
                            }
                        ]
                    }
                )
                print(f"Endpoint '{endpoint_name}' created serving version {model_version_to_serve}.")
            else:
                raise e
        
        # Wait for the endpoint to be ready (optional, but good for automation pipelines)
        print("Waiting for endpoint to become ready...")
        TIMEOUT_SECONDS = 20 * 60 # 20 minutes
        POLL_INTERVAL_SECONDS = 10 # Check every 10 seconds
        start_time = time.time() # Record the start time
        status = ""
        while True:
            try:
                endpoint = client.get_endpoint(endpoint_name)
                status = endpoint.state.ready
                update_status = endpoint.state.config_update
            except Exception as e:
                # Handle cases where getting endpoint status itself might fail (e.g., transient network issue)
                print(f"Error getting endpoint status: {e}")
                status = "UNKNOWN" # Set status to something that won't make it READY

            print(f"Endpoint status: {status} (Elapsed: {int(time.time() - start_time)}s)")
            

            if status == "READY" and update_status == "NOT_UPDATING":
                print(f"\nEndpoint '{endpoint_name}' is READY.")
                break # Exit loop if ready
            
            # Check for timeout *after* checking if it's READY
            if time.time() - start_time > TIMEOUT_SECONDS:
                print(f"\nTimeout: Endpoint '{endpoint_name}' did not become READY within {TIMEOUT_SECONDS} seconds.")
                print(f"Current status: {status}")
                # client.delete_endpoint(endpoint=endpoint_name)
                # Forcefully stop the script by raising an exception
                raise TimeoutError(f"Endpoint '{endpoint_name}' failed to become ready within {TIMEOUT_SECONDS} seconds. Current status: {status}")

            sleep(POLL_INTERVAL_SECONDS) # Wait before checking again

        print(f"Endpoint URL: {DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations")
        print("You can find more details in the Databricks UI under 'Serving'.")
    except TimeoutError as te:
        print(f"Operation for {endpoint_name} timed out: {te}. Moving to next endpoint.")# This catch block specifically handles the TimeoutError, logs it,
    except Exception as e:
        print(f"An unexpected error occurred for {endpoint_name}: {e}. Moving to next endpoint.")
        # Catch any other unexpected errors during the process for this endpoint


# COMMAND ----------

#Testing the registered model and input payload sample.
import mlflow.models
from mlflow.models.utils import convert_input_example_to_serving_input
serving_input_example_dict = [{
    "Client": "rcc",
    "FX_Instrument_Type": "Spot",
    "BUnit": "2",
    "Cpty": "RBST",
    "PrimaryCurr": "EUR",
    "BuyCurr": "EUR",
    "SellCurr": "USD",
    "BuyAmount": 1000.0,
    "SellAmount": 800.0,
    "SpotRate": 1.25,
    "ForwardPoints": 0.0,
    "ActivationDate": "2007-05-29",
    "MaturityDate": "2007-06-4",
    "UniqueId": "FX123456789"
}]

serving_input = convert_input_example_to_serving_input(serving_input_example_dict)
client_tracking = mlflow.tracking.MlflowClient()
for container, info in registered_models_info.items():
    try:
        register_model = info['model_name']
        latest_version = client_tracking.get_latest_versions(register_model)
        if latest_version:
            version = latest_version[0].version
        else:
            raise ValueError(f"No version found for model '{register_model}'")
        model_uri = f"models:/{register_model}/{version}"
        # This will print the actual error if validation fails
        mlflow.models.validate_serving_input(
            model_uri=model_uri,
            serving_input=serving_input
        )
        print("Explicit serving input validation successful!")
    except Exception as e:
        print(f"Explicit serving input validation FAILED: {e}")
