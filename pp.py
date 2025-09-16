import builtins
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import pickle
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from datetime import datetime
from pydantic import BaseModel, ValidationError
import mlflow
from typing import List, Dict, Union, Any
import time
import sys
from azure.identity import ClientSecretCredential
from databricks.sql import connect

# Required environment variables
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
WORKSPACE_FQDN = os.getenv("WORKSPACE_FQDN")
SQL_WAREHOUSE_HTTP_PATH = os.getenv("SQL_WAREHOUSE_HTTP_PATH")
AZURE_SP_CLIENT_ID = os.getenv("AZURE_SP_CLIENT_ID")
AZURE_SP_CLIENT_SECRET = os.getenv("AZURE_SP_CLIENT_SECRET")  # MUST be injected at runtime
LOG_TABLE_UC_NAME = os.getenv("LOG_TABLE_UC_NAME")
DATABRICKS_SCOPE = os.getenv("DATABRICKS_SCOPE")


if not AZURE_SP_CLIENT_SECRET:
    raise RuntimeError("AZURE_SP_CLIENT_SECRET is not set. Configure it as an environment variable.")


def _get_access_token() -> str:
    cred = ClientSecretCredential(
        client_id=AZURE_SP_CLIENT_ID,
        client_secret=AZURE_SP_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID
    )
    return cred.get_token(DATABRICKS_SCOPE).token

from datetime import datetime, timezone
import json

def _ts(x):
    """Convert epoch seconds / ISO string / datetime to tz-aware datetime (UTC) or None."""
    if x is None or x == '':
        return None
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace('Z', '+00:00'))
        except Exception:
            return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    return None


def _to_json(payload) -> str:
    """Safely JSON-serialize dict/str/DataFrame/ndarray/etc."""
    try:
        if isinstance(payload, pd.DataFrame):
            payload = json.loads(payload.to_json(orient="records"))
        return json.dumps(payload, default=str)
    except Exception:
        try:
            return json.dumps(str(payload))
        except Exception:
            return '"<unserializable>"'

def log_event(
    log_id: int,
    log_start_time=None,
    log_end_time=None,
    model_name: str = None,
    model_version: str = None,
    request_payload=None,
    response_payload=None,
    api_status: str = None,
    inference_start_time=None,
    inference_end_time=None,
    autoencoder_start_time=None,
    autoencoder_end_time=None,
):
    """
    Inserts one row into api_log table with 12 columns:
      (log_id, log_start_time, log_end_time, model_name, model_version,
       request_payload, response_payload, api_status,
       inference_start_time, inference_end_time, autoencoder_start_time, autoencoder_end_time)

    Caller must provide all values (or None for missing).
    """
    try:
        token = _get_access_token()
        with connect(
            server_hostname=WORKSPACE_FQDN,
            http_path=SQL_WAREHOUSE_HTTP_PATH,
            access_token=token
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {LOG_TABLE_UC_NAME}
                        (log_id,
                         log_start_time,
                         log_end_time,
                         model_name,
                         model_version,
                         request_payload,
                         response_payload,
                         api_status,
                         inference_start_time,
                         inference_end_time,
                         autoencoder_start_time,
                         autoencoder_end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_id,_ts(log_start_time),_ts(log_end_time),model_name,model_version,_to_json(request_payload),_to_json(response_payload),
                        api_status,_ts(inference_start_time),_ts(inference_end_time),_ts(autoencoder_start_time),_ts(autoencoder_end_time),
                    ),
                )
                # conn.commit() if needed
    except Exception as e:
        print(f"[log_event] WARN: failed to write log: {e}")


# ---- +++ End of logging function +++ ------------------


from mlflow.tracking import MlflowClient

def get_model_version(model_name: str, stage: str = None):
    client = MlflowClient()
    if stage:
        latest = client.get_latest_versions(model_name, [stage])
        return latest[0].version if latest else "unknown"
    # Or fetch all versions and pick the highest
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(int(v.version) for v in versions) if versions else "unknown"


# --- Pydantic Model ---
class ValidateParams(BaseModel):
    '''
    Validate the input params.
    '''
    Client: str
    FX_Instrument_Type: str
    BUnit: str
    Cpty: str
    PrimaryCurr: str
    BuyCurr: str
    SellCurr: str
    BuyAmount: float
    SellAmount: float
    SpotRate: float
    ForwardPoints: float
    DealDate: str
    MaturityDate: str
    UniqueId: str
    ContLeg: int = None
# --- End Pydantic Model ---

import pickle
import builtins
import importlib
from sklearn.preprocessing import MinMaxScaler  # Explicit import

class _RestrictedUnpickler(pickle.Unpickler):
    """Allow harmless built-ins + selected sklearn and numpy classes."""

    _ALLOWED = {"dict", "list", "tuple", "set", "str", "int", "float", "bool", "bytes","slice"}
    _ALLOW_MODULE_PREFIXES = ("sklearn", "numpy","pandas")

    # Explicitly map internal references to safe classes
    _CLASS_REGISTRY = {
        ("sklearn.preprocessing._data", "MinMaxScaler"): MinMaxScaler,
    }

    def find_class(self, module, name):
        if (module == "builtins" and name in self._ALLOWED):
            return getattr(builtins, name)

        if (module, name) in self._CLASS_REGISTRY:
            return self._CLASS_REGISTRY[(module, name)]

        if module.startswith(self._ALLOW_MODULE_PREFIXES):
            try:
                mod = importlib.import_module(module)
                return getattr(mod, name)
            except (ImportError, AttributeError):
                pass

        raise pickle.UnpicklingError(f"Forbidden global during unpickling: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent IDs are forbidden")

def _safe_pickle_load(file_obj):
    return _RestrictedUnpickler(file_obj).load()

class KerasAnomalyDetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the Keras Autoencoder model and all preprocessors/scalers.
        Assumes these are saved as artifacts in the MLflow run.
        """
        
        actual_keras_file_path = context.artifacts["keras_autoencoder_path"]
        self.model = tf.keras.models.load_model(actual_keras_file_path)

        self.allowed_client_for_this_endpoint = context.model_config.get("allowed_client", "default")

        # --- SAFE LOAD for scalers ---
        with open(context.artifacts["scalers_path"], "rb") as f:
            scalers_dict = _safe_pickle_load(f)
        self.load_scalers = scalers_dict[self.allowed_client_for_this_endpoint]

        # --- SAFE LOAD for year_gap_data (same as scalers) ---
        with open(context.artifacts["year_gap_data_path"], "rb") as f:
            self.year_gap_data = _safe_pickle_load(f)

        # Pre-extract uniques
        self.unique_BU, self.unique_cpty, self.unique_primarycurr = self._get_uniques(
            self.load_scalers['grouped_scalers']
        )


    # --- Helper methods for `inference` ---
    def _get_column_types(self, df: pd.DataFrame):
        categorical_columns = []
        numeric_columns = []
        for feature in df.columns:
            if df[feature].dtype == object: # Also check for categorical dtype
                categorical_columns.append(feature)
            elif df[feature].dtype in (float, int) :
                numeric_columns.append(feature)
        return categorical_columns, numeric_columns

    def _one_hot(self, df: pd.DataFrame, encoder: OneHotEncoder):
        # The encoder is provided, so no fitting here
        encoded_data = encoder.transform(df)
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
        return encoded_df

    def _scale(self, df: pd.DataFrame, scaler: MinMaxScaler):
        # The scaler is provided, so no fitting here
        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
        return scaled_df

    def _autoencode(self, input_df_list: list[pd.DataFrame]):
        # The model is now self.model
        features = pd.concat(input_df_list, axis=1)
        # Keras predict returns a list of arrays for multi-output models
        reconstructed_features_raw = self.model.predict(input_df_list)
        
        # Ensure reconstructed_features_raw is a list even for single output, for consistency
        if not isinstance(reconstructed_features_raw, list):
            reconstructed_features_raw = [reconstructed_features_raw]

        reconstructed_df = pd.DataFrame()
        reconstructed_normalized_df = pd.DataFrame()
        
        feature_len = len(input_df_list) # This represents number of distinct inputs to autoencoder
        # The last one is numeric, others are one-hot encoded categorical.
        numeric_idx = feature_len - 1 # Assuming numeric features are last input to autoencoder

        for i in range(feature_len):
            if i != numeric_idx:
                # For categorical, apply argmax to get one-hot from softmax-like output
                df1 = pd.DataFrame(np.array([np.where(l == max(l), 1.0, 0.0) for l in reconstructed_features_raw[i]]))
                df2 = pd.DataFrame(np.array(reconstructed_features_raw[i]))
            else:
                # For numeric, direct output
                df1 = pd.DataFrame(reconstructed_features_raw[i])
                df2 = pd.DataFrame(reconstructed_features_raw[i])
            
            reconstructed_normalized_df = pd.concat([reconstructed_normalized_df, df1], axis=1)
            reconstructed_df = pd.concat([reconstructed_df, df2], axis=1)

        reconstructed_normalized_df.columns = features.columns
        reconstructed_df.columns = features.columns
        
        return features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df

    def _remove_list(self, original_list: list, remove_list: list) -> list:
        for i in remove_list:
            original_list.remove(i)
        return original_list

    def _face_value(self, df:pd.DataFrame) -> pd.DataFrame:
        df["FaceValue"] = np.nan
        if df.PrimaryCurr == df.BuyCurr:
            df["FaceValue"]=np.abs(df.BuyAmount)
        elif df.PrimaryCurr == df.SellCurr:
            df["FaceValue"]=np.abs(df.SellAmount)
        return df

    def _get_uniques(self, grouped_scalers):
        unique_BU = set()
        unique_cpty = set()
        unique_primarycurr = set()
        for i in grouped_scalers:
            BUnit, Cpty, PrimaryCurr = i
            unique_BU.add(BUnit)
            unique_cpty.add(Cpty)
            unique_primarycurr.add(PrimaryCurr)
        return unique_BU, unique_cpty, unique_primarycurr

    def _check_missing_group(self, data: pd.DataFrame):
        missed = 0
        BU = data['BUnit'].values[0]
        CPTY = data['Cpty'].values[0]
        PC = data['PrimaryCurr'].values[0]
        #Check for missing whole group combination
        group = (BU,CPTY,PC)
        if not group in self.load_scalers.keys():
            missed = 1
            message =  f"This is the first FX contract with this busniness unit, counter party & primary currency"
            return missed, message
        missing_BU = BU not in self.unique_BU
        missing_Cpty = CPTY not in self.unique_cpty
        missing_PC = PC not in self.unique_primarycurr
        missing_BU_Cpty = missing_BU and missing_Cpty
        missing_Cpty_PC = missing_Cpty and missing_PC
        missing_BU_PC = missing_BU and missing_PC
        missing_all = missing_BU and missing_Cpty and missing_PC
        
        if missing_all:
            missed = 1
            message =  f"This is the first FX contract with this busniness unit, counter party & primary currency"
        elif missing_BU_Cpty:
            missed = 1
            message =  f"This is the first FX contract with this busniness unit & counterparty"
        elif missing_Cpty_PC:
            missed = 1
            message =  f"This is the first FX contract with this counterparty and primary currency"
        elif missing_BU_PC:
            missed = 1
            message =  f"This is the first FX contract with this busniness unit and primary currency "
        elif missing_BU:
            missed = 1
            message =  f"This is the first FX contract with this business unit"
        elif missing_Cpty:
            missed = 1
            message =  f"This is the first FX contract with this counterparty"
        elif missing_PC:
            missed = 1
            message =  f"This is the first FX contract with this primary currency."
        else:
            message = 'No missing data'
        return missed, message

    def _check_currency(self, data: pd.DataFrame):
        new = 0
        CP = data['Cpty'].values[0]
        BuyCurr = data['BuyCurr'].values[0]
        SellCurr = data['SellCurr'].values[0]
        if CP not in self.load_scalers['cpty_group']:
            # Handle case where Cpty itself is new
            new = 1
            message = f"This is the first FX contract with this counterparty."
            return new, message

        trained_cptys_for_cp = self.load_scalers['cpty_group'][CP]
        missing_BuyCurr = BuyCurr not in trained_cptys_for_cp.get('buy', {})
        missing_SellCurr = SellCurr not in trained_cptys_for_cp.get('sell', {})
        
        missing_BuySell = missing_BuyCurr and missing_SellCurr
        if missing_BuySell or missing_BuyCurr or missing_SellCurr:
            new = 1
            message = f"This currency pair has not been traded before for this counter party."
        else:
            message = "No new data"
        return new, message
    
    def _compare_rows(self,df1: pd.DataFrame, df2: pd.DataFrame, cols=None):
        if cols is None:
            cols = ["BUnit", "Cpty", "BuyCurr", "SellCurr", "DealDate"]
        row = df2.iloc[0]
        return (df1[cols] == row[cols].values).all(axis=1).any()

    def _inference_pipeline(self, data: pd.DataFrame, client_name: str = 'rcc'):
        autoencoder_start_time=None
        autoencoder_end_time=None
        col_list = ['FX_Instrument_Type', 'BUnit', 'Cpty','DealDate',
                    'MaturityDate', 'PrimaryCurr','BuyAmount',
                    'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']
        categories_in_numerics = ['BUnit'] # Your code casts BUnit to str then to category
        
        # logger.info(f"Preprocessing started for client: {client_name}.")
        data = data[col_list].copy()
        data.rename(columns={'TranType':'FX_Instrument_Type'}, inplace=True)
        data.rename(columns={'InstrumentType':'FX_Instrument_Type'}, inplace=True)
        data.rename(columns={'Instrument':'FX_Instrument_Type'}, inplace=True)
        data.rename(columns={'ActivationDate':'DealDate'}, inplace=True)
        data[['MaturityDate', 'DealDate']] = data[['MaturityDate', 'DealDate']].apply(pd.to_datetime)
        data[['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints']] = data[['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints']].astype(float)
        # mapping = {'FXFORWARD': 'FX Forward','FXSP_MODEL': 'FX SP_Model',
        #     'FXSPOT': 'FX Spot','FXSWAP': 'FX Swap'}
        # data['FX_Instrument_Type'] = data['FX_Instrument_Type'].map(mapping)
        # Load data with year_gaps
        year_gaps_df = self.year_gap_data.copy()
        is_year_gap = self._compare_rows(year_gaps_df,data)
        if is_year_gap:
            message =  "This currency pair has not been traded in the previous 12 months"
            return autoencoder_start_time,autoencoder_end_time,message,'', '', '', '', ''
        
        data['Is_weekend_date'] = data.DealDate.apply(lambda x: x.date().isoweekday())
        data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x < 6 else 1)
        data['TDays'] = (data.MaturityDate - data.DealDate).dt.days

        data[categories_in_numerics] = data[categories_in_numerics].astype(str)
        categorical_columns, numeric_columns = self._get_column_types(data)
        # Ensure consistent dtypes for concatenation, especially for string/object
        data = pd.concat((data[categorical_columns].astype('category'), data[numeric_columns].astype(float)), axis=1)
        
        data = data.apply(self._face_value, axis=1)
        data.fillna(0, inplace=True) # Fill missing values after face_value

        # logger.info(f"Checking Business Logics.")
        missing, message = self._check_missing_group(data)
        if missing:
            return autoencoder_start_time,autoencoder_end_time,message, '', '', '', '',''
        new, message = self._check_currency(data)
        if new:
            return autoencoder_start_time,autoencoder_end_time,message, '', '', '', '',''
        fv_lower = self.load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['lower']
        fv_upper = self.load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['upper']
        tdays_lower = self.load_scalers['tdays_scalers'][(data.iloc[0]['FX_Instrument_Type'],)]['lower']
        tdays_upper = self.load_scalers['tdays_scalers'][(data.iloc[0]['FX_Instrument_Type'],)]['upper']
        fv_tdays_dict = {'facevalue_lower':fv_lower,
                        'facevalue_upper':fv_upper,
                        'tdays_lower':tdays_lower,
                        'tdays_upper':tdays_upper,
                        'actual_facevalue':data['FaceValue'].iloc[0],
                        'actual_tdays':data['TDays'].iloc[0]}
        
        # Apply grouped scalers
        # Ensure the column used for lookup (e.g., 'BUnit', 'Cpty', 'PrimaryCurr') are in the correct format (e.g., string)
        data["FaceValue"] = data["FaceValue"].apply(lambda x: 0 if fv_lower <= x <= fv_upper else 1)
        data["TDays"] = data["TDays"].apply(lambda x: 0 if tdays_lower <= x <= tdays_upper else 1)
        
        # logger.info(f"Applying scaling techniques.")
        # Make a copy of numeric_columns to avoid modifying the original list
        current_numeric_columns = list(numeric_columns) 
        current_numeric_columns = self._remove_list(current_numeric_columns, ['TDays'])

        # Use the loaded one-hot encoder and min-max scaler
        cat_data = self._one_hot(data[categorical_columns], self.load_scalers['ohe'])
        num_data = self._scale(data[current_numeric_columns], self.load_scalers['mms'])
        
        num_data['FaceValue'] = data['FaceValue'].values
        num_data['TDays'] = data['TDays'].values
        
        # Split data to multi-input format as per trained model
        processed_df_list = []
        for category in categorical_columns:
            category_cols = [x for x in cat_data.columns if x.startswith(category)]
            # Ensure the order of columns in category_df matches what the autoencoder was trained on
            # This is critical for multi-input Keras models
            category_df = cat_data[category_cols]
            processed_df_list.append(category_df)
        processed_df_list.append(num_data) # Numeric features as the last input

        # logger.info(f"Feeding data to encoder model.")
        autoencoder_start_time = time.time()
        features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df = self._autoencode(processed_df_list)
        autoencoder_end_time = time.time() 
        scores = pd.DataFrame(np.sqrt(np.mean(np.square(features - reconstructed_df), axis=1)), columns=['RMSE'])
        
        return autoencoder_start_time,autoencoder_end_time,scores, features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df,fv_tdays_dict

    def _get_condition_filters(self, df_deviation:pd.DataFrame):
        '''
        It filter the encoder model predictions by applying the thresholds.
        Input:
            df_deviation - DataFrame created with differnce in actual and model predictions.
        Output:
            Deviated_Features - contains the features which are highly deviated
            counts - anomaly/non-anomaly counts.
        '''
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
    
    def _get_filtered_data(self, features:pd.DataFrame, reconstructed_df:pd.DataFrame) -> dict:
        '''
        Creates the final format to feed to LLM.
        Input:
            -features: Raw features after transformation.
            -reconstructed_df: model prediction after transformations.
        Output:
            filtered_data: highly deviated features list along with BUnit and Cpty in json.
        '''
        df_deviation = features-reconstructed_df
        anomaly_count , non_anomaly_count, Deviated_Features = self._get_condition_filters(df_deviation)
        actual = [col for col in features.columns if (col.startswith('BUnit') or col.startswith('Cpty')) and (features[col] == 1).any()]
        Bunit = actual[0].split('_')
        Cpty = actual[1].split('_')
        filtered_data = {Bunit[0]:Bunit[1], Cpty[0]:Cpty[1], 'Deviated_Features':Deviated_Features}
        return filtered_data

    def _get_llm_input(context,input_data, df_deviation,fv_tdays_dict):
        llm_input = f'\nData:\n{input_data} \n' + f'\nDeviated features:\n{df_deviation.to_dict()} \n'\
            f"\nfv_tdays_dict : {fv_tdays_dict} \n" + f"\nContext:{context}"
        return llm_input


    def predict(self, context, model_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predicts anomalies for a list of input dictionaries.

        Args:
            context: The MLflow context.
            model_input: A list of dictionaries, where each dictionary represents
                         a single data point for prediction.

        Returns:
            A list of dictionaries. Each dictionary will contain either the
            prediction result for a successfully processed input, or an
            error message and status for a failed input (e.g., "status": "error").
            If the `model_input` itself is invalid (e.g., not a list), an
            exception will be raised, leading to an HTTP 500 (or similar) error
            by the serving environment.
        """
        # If model_input is not a list, an error will be raised by this check.
        # MLflow's pyfunc server will typically catch this and return a 500 error.
        inference_start_time = time.time()
        log_start= time.time()
        if not isinstance(model_input, list):
            item_result = {
                    "status_code": 500,
                    "error": "Input format is not List of dictionaries",
                }
            # ------- +++ logging event +++ ----
           
            log_event(
                log_id=None,
                log_start_time=log_start,
                log_end_time=time.time(),
                model_name=None,
                model_version=None,
                request_payload=model_input,
                response_payload=item_result,
                api_status="FAILURE",
                inference_start_time=None,
                inference_end_time=time.time(),
                autoencoder_start_time=None,
                autoencoder_end_time=None
            )
             # ------- +++ logging event +++ ----
            return [item_result]
        
        if not model_input:
            # If an empty list is provided, return an empty list of results.
            # The HTTP status code will typically be 200 OK.
            item_result = {
                    "status_code": 200,
                    "message": "Send valid input.[non empty]",
                }
            # ------- +++ logging event +++ ----
            log_event(
                log_id=None,
                log_start_time=log_start,
                log_end_time=time.time(),
                model_name=None,
                model_version=None,
                request_payload=model_input,
                response_payload=item_result,
                api_status="OK-EMPTY",
                inference_start_time=None,
                inference_end_time=time.time(),
                autoencoder_start_time=None,
                autoencoder_end_time=None
            )
             # ------- +++ logging event +++ ----
            return [item_result]

        results = []
        cpty_threshold, threshold_1,threshold_2 = 0.995, 0.97, 0.96
        
        callsign='fx_anomaly_prediction_'+self.allowed_client_for_this_endpoint

        for _, input_dict in enumerate(model_input):
            item_result = {} # Store result for the current item
            ID=input_dict['UniqueId']
            try:
                validated_data = ValidateParams(**input_dict)
                data_dict = validated_data.model_dump() # Use .model_dump() for Pydantic v2+
                expected_client = self.allowed_client_for_this_endpoint
                if expected_client != data_dict['Client']:
                    raise ValueError(f"Invalid client name for this endpoint. Expected: {expected_client}, Got: {data_dict['Client']}")

                # Convert date strings to datetime objects
                data_dict['DealDate'] = datetime.strptime(data_dict['DealDate'], "%Y-%m-%d")
                data_dict['MaturityDate'] = datetime.strptime(data_dict['MaturityDate'], "%Y-%m-%d")

                client_dict = {'Client': data_dict['Client'], 'UniqueId': data_dict['UniqueId']}
                client_dict['Anomaly'] = 'No' # Default to No
                if "ContLeg" in data_dict.keys():
                    client_dict['ContLeg'] = data_dict['ContLeg']

                inference_data = {
                    k: v for k, v in data_dict.items()
                    if k not in {'Client', 'UniqueId', 'ContLeg'}
                }
                inference_data = pd.DataFrame([inference_data], index=[0])
               
                
                # Run the core inference pipeline
                autoencoder_start_time=None
                autoencoder_end_time=None

                autoencoder_start_time,autoencoder_end_time,result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df,fv_tdays_dict = \
                    self._inference_pipeline(inference_data, client_name=client_dict['Client'].lower())
                
                
                if isinstance(result, str): # Anomaly detected by business rules
                    client_dict['Anomaly'] = 'Yes'
                    client_dict['Reason'] = result
                    #model_details['Inference_time'] = Inference_time
                    #model_details['autencoder_time'] = autencoder_time
                else: # Proceed with autoencoder anomaly detection
                    df_deviation_all = features-reconstructed_df
                    df_deviation_all = df_deviation_all.round(3)
                    df_dev_bu_cpty = df_deviation_all.loc[:,df_deviation_all.columns.str.startswith(('Cpty'))]
                    if df_dev_bu_cpty[df_dev_bu_cpty>cpty_threshold].any().sum()>=1:#changed
                        client_dict['Anomaly'] = 'Yes'
                        client_dict['Reason'] = 'The deal is anomalous.'
                        #model_details['Inference_time'] = Inference_time
                    df_deviation = df_deviation_all.loc[:, ~df_deviation_all.columns.str.startswith(('BUnit', 'Cpty'))]    
                    if df_deviation[df_deviation>threshold_1].any().sum()>=1:
                        client_dict['Anomaly'] = 'Yes'
                        client_dict['Reason'] = 'The deal is anomalous.'
                        #model_details['Inference_time'] = Inference_time
                    elif df_deviation[df_deviation>threshold_2].any().sum()>2:
                        client_dict['Anomaly'] = 'Yes'
                        client_dict['Reason'] = 'The deal is anomalous.'
                        #model_details['Inference_time'] = Inference_time
                    else:
                        client_dict['Anomaly'] = 'No'
                        client_dict['Reason'] = '' #Integrity wants it blank in case of non-anomaly
                        #model_details['Inference_time'] = Inference_time
                # Add success status to the item result
                item_result = {**client_dict, "status_code": 200} # Merging client_dict and status
               
                #inference_end_time = time.time() 
                # ------- +++ logging event +++ ----
                
                log_event(
                    log_id=ID,
                    log_start_time=log_start,
                    log_end_time=time.time(),
                    model_name=self.allowed_client_for_this_endpoint,
                    model_version=get_model_version(callsign),
                    request_payload=model_input,
                    response_payload=item_result,
                    api_status="OK",
                    inference_start_time=inference_start_time,
                    inference_end_time=time.time(),
                    autoencoder_start_time=autoencoder_start_time,
                    autoencoder_end_time=autoencoder_end_time
                )
                # ------- +++ logging event +++ ----

            except ValidationError as e:
                item_result = {
                    "status_code": 400,
                    "error": f"Input validation error: {e}",
                }
                # ------- +++ logging event +++ ----
                
                log_event(
                    log_id=ID,
                    log_start_time=log_start,
                    log_end_time=time.time(),
                    model_name=self.allowed_client_for_this_endpoint,
                    model_version=get_model_version(callsign),
                    request_payload=model_input,
                    response_payload=item_result,
                    api_status="FAILURE",
                    inference_start_time=inference_start_time,
                    inference_end_time=time.time(),
                    autoencoder_start_time=None,
                    autoencoder_end_time=None
                )
                # ------- +++ logging event +++ ----

            except KeyError as e:
                item_result = {
                    "status_code": 400,
                    "error": f"Got unexpected key '{e}'",
                }
                # ------- +++ logging event +++ ----
                log_event(
                    log_id=ID,
                    log_start_time=log_start,
                    log_end_time=time.time(),
                    model_name=self.allowed_client_for_this_endpoint,
                    model_version=get_model_version(callsign),
                    request_payload=model_input,
                    response_payload=item_result,
                    api_status="FAILURE",
                    inference_start_time=inference_start_time,
                    inference_end_time=time.time(),
                    autoencoder_start_time=None,
                    autoencoder_end_time=None
                )
                # ------- +++ logging event +++ ----
            except Exception as e:
                # For unexpected errors during processing of an item, capture the error
                # but don't re-raise immediately, allow other items to process.
                item_result = {
                    "status_code": 500,
                    "error": f"An unexpected error occurred during processing: {str(e)}",
                }
                 # ------- +++ logging event +++ ----
                log_event(
                    log_id=ID,
                    log_start_time=log_start,
                    log_end_time=time.time(),
                    model_name=self.allowed_client_for_this_endpoint,
                    model_version=get_model_version(callsign),
                    request_payload=model_input,
                    response_payload=item_result,
                    api_status="FAILURE",
                    inference_start_time=inference_start_time,
                    inference_end_time=time.time(),
                    autoencoder_start_time=None,
                    autoencoder_end_time=None
                )
                # ------- +++ logging event +++ ----

            results.append(item_result)

        return results
