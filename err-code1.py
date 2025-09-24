import os, json, time
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import mlflow
from typing import List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from azure.identity import ClientSecretCredential
from databricks.sql import connect

# --- New modular imports ---
from utils.serializers import to_utc as _ts, to_json_safely as _to_json
from utils.safe_pickle import safe_load as _safe_pickle_load
from utils.artifacts import load_artifacts_from_context
from utils.rules import compare_rows, check_missing_group, check_currency
from utils.preprocess import _get_column_types,_one_hot,_scale,_remove_list,_face_value,_get_uniques

# ===== Env Vars =====
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
WORKSPACE_FQDN = os.getenv("WORKSPACE_FQDN")
SQL_WAREHOUSE_HTTP_PATH = os.getenv("SQL_WAREHOUSE_HTTP_PATH")
AZURE_SP_CLIENT_ID = os.getenv("AZURE_SP_CLIENT_ID")
AZURE_SP_CLIENT_SECRET = os.getenv("AZURE_SP_CLIENT_SECRET")
LOG_TABLE_UC_NAME = os.getenv("LOG_TABLE_UC_NAME")
DATABRICKS_SCOPE = os.getenv("DATABRICKS_SCOPE")

if not AZURE_SP_CLIENT_SECRET:
    raise RuntimeError("AZURE_SP_CLIENT_SECRET is not set. Configure it as an environment variable.")

# ===== Small helper for standardized api_status =====
def _api_status_from_code(code: int, tag: str | None = None) -> str:
    base = {
        200: "OK",
        400: "BAD_REQUEST",
        403: "FORBIDDEN",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_ERROR",
    }.get(code, "UNKNOWN")
    return f"{base}{'-' + tag if tag else ''}"

# ===== Auth & Logging =====
def _get_access_token() -> str:
    cred = ClientSecretCredential(
        client_id=AZURE_SP_CLIENT_ID,
        client_secret=AZURE_SP_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID
    )
    return cred.get_token(DATABRICKS_SCOPE).token

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
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN, http_path=SQL_WAREHOUSE_HTTP_PATH, access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {LOG_TABLE_UC_NAME}
                        (log_id, log_start_time, log_end_time, model_name, model_version,
                         request_payload, response_payload, api_status,
                         inference_start_time, inference_end_time, autoencoder_start_time, autoencoder_end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_id, _ts(log_start_time), _ts(log_end_time), model_name, model_version,
                        _to_json(request_payload), _to_json(response_payload), api_status,
                        _ts(inference_start_time), _ts(inference_end_time),
                        _ts(autoencoder_start_time), _ts(autoencoder_end_time),
                    ),
                )
    except Exception as e:
        print(f"[log_event] WARN: failed to write log: {e}")

# ===== MLflow Registry helper =====
from mlflow.tracking import MlflowClient
def get_model_version(model_name: str, stage: str = None):
    client = MlflowClient()
    if stage:
        latest = client.get_latest_versions(model_name, [stage])
        return latest[0].version if latest else "unknown"
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(int(v.version) for v in versions) if versions else "unknown"

# ===== Request schema =====
class ValidateParams(BaseModel):
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
    ContLeg: int | None = None

class KerasAnomalyDetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.allowed_client_for_this_endpoint = context.model_config.get("allowed_client", "default")
        arts = load_artifacts_from_context(context)
        scalers_dict = arts.scalers_dict
        if self.allowed_client_for_this_endpoint not in scalers_dict:
            raise KeyError(f"Client '{self.allowed_client_for_this_endpoint}' not found in scalers artifact")
        self.load_scalers = scalers_dict[self.allowed_client_for_this_endpoint]
        self.year_gap_data = arts.year_gap_data
        self.model = arts.model
        self.unique_BU, self.unique_cpty, self.unique_primarycurr = self._get_uniques(
            self.load_scalers['grouped_scalers']
        )

    def _autoencode(self, input_df_list: list[pd.DataFrame]):
        features = pd.concat(input_df_list, axis=1)
        reconstructed_features_raw = self.model.predict(input_df_list)
        if not isinstance(reconstructed_features_raw, list):
            reconstructed_features_raw = [reconstructed_features_raw]
        reconstructed_df = pd.DataFrame()
        reconstructed_normalized_df = pd.DataFrame()
        feature_len = len(input_df_list)
        numeric_idx = feature_len - 1
        for i in range(feature_len):
            if i != numeric_idx:
                df1 = pd.DataFrame(np.array([np.where(l == max(l), 1.0, 0.0) for l in reconstructed_features_raw[i]]))
                df2 = pd.DataFrame(np.array(reconstructed_features_raw[i]))
            else:
                df1 = pd.DataFrame(reconstructed_features_raw[i])
                df2 = pd.DataFrame(reconstructed_features_raw[i])
            reconstructed_normalized_df = pd.concat([reconstructed_normalized_df, df1], axis=1)
            reconstructed_df = pd.concat([reconstructed_df, df2], axis=1)
        reconstructed_normalized_df.columns = features.columns
        reconstructed_df.columns = features.columns
        return features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df

    def _inference_pipeline(self, data: pd.DataFrame, client_name: str = 'rcc'):
        autoencoder_start_time=None; autoencoder_end_time=None
        col_list = ['FX_Instrument_Type','BUnit','Cpty','DealDate','MaturityDate','PrimaryCurr',
                    'BuyAmount','SellAmount','BuyCurr','SellCurr','SpotRate','ForwardPoints']
        categories_in_numerics = ['BUnit']

        data = data[col_list].copy()
        data.rename(columns={'TranType':'FX_Instrument_Type','InstrumentType':'FX_Instrument_Type','Instrument':'FX_Instrument_Type','ActivationDate':'DealDate'}, inplace=True)
        data[['MaturityDate','DealDate']] = data[['MaturityDate','DealDate']].apply(pd.to_datetime)
        data[['BuyAmount','SellAmount','SpotRate','ForwardPoints']] = data[['BuyAmount','SellAmount','SpotRate','ForwardPoints']].astype(float)

        year_gaps_df = self.year_gap_data.copy()
        if isinstance(year_gaps_df, pd.DataFrame) and not year_gaps_df.empty and compare_rows(year_gaps_df, data):
            message = "This currency pair has not been traded in the previous 12 months"
            return autoencoder_start_time, autoencoder_end_time, message, '', '', '', '', ''

        data['Is_weekend_date'] = data.DealDate.apply(lambda x: x.date().isoweekday())
        data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x < 6 else 1)
        data['TDays'] = (data.MaturityDate - data.DealDate).dt.days
        data[categories_in_numerics] = data[categories_in_numerics].astype(str)

        categorical_columns, numeric_columns = self._get_column_types(data)
        data = pd.concat((data[categorical_columns].astype('category'), data[numeric_columns].astype(float)), axis=1)
        data = data.apply(self._face_value, axis=1)
        data.fillna(0, inplace=True)

        missed, msg = check_missing_group(data, self.load_scalers, self.unique_BU, self.unique_cpty, self.unique_primarycurr)
        if missed:
            return autoencoder_start_time, autoencoder_end_time, msg, '', '', '', '', ''
        new, msg = check_currency(data, self.load_scalers)
        if new:
            return autoencoder_start_time, autoencoder_end_time, msg, '', '', '', '', ''

        fv_lower = self.load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['lower']
        fv_upper = self.load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['upper']
        tdays_lower = self.load_scalers['tdays_scalers'][(data.iloc[0]['FX_Instrument_Type'],)]['lower']
        tdays_upper = self.load_scalers['tdays_scalers'][(data.iloc[0]['FX_Instrument_Type'],)]['upper']

        fv_tdays_dict = {
            'facevalue_lower': fv_lower, 'facevalue_upper': fv_upper,
            'tdays_lower': tdays_lower, 'tdays_upper': tdays_upper,
            'actual_facevalue': data['FaceValue'].iloc[0], 'actual_tdays': data['TDays'].iloc[0]
        }

        data["FaceValue"] = data["FaceValue"].apply(lambda x: 0 if fv_lower <= x <= fv_upper else 1)
        data["TDays"] = data["TDays"].apply(lambda x: 0 if tdays_lower <= x <= tdays_upper else 1)

        current_numeric_columns = list(numeric_columns)
        current_numeric_columns = self._remove_list(current_numeric_columns, ['TDays'])

        cat_data = self._one_hot(data[categorical_columns], self.load_scalers['ohe'])
        num_data = self._scale(data[current_numeric_columns], self.load_scalers['mms'])
        num_data['FaceValue'] = data['FaceValue'].values
        num_data['TDays'] = data['TDays'].values

        processed_df_list = []
        for category in categorical_columns:
            category_cols = [x for x in cat_data.columns if x.startswith(category)]
            category_df = cat_data[category_cols]
            processed_df_list.append(category_df)
        processed_df_list.append(num_data)

        autoencoder_start_time = time.time()
        features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df = self._autoencode(processed_df_list)
        autoencoder_end_time = time.time()
        scores = pd.DataFrame(np.sqrt(np.mean(np.square(features - reconstructed_df), axis=1)), columns=['RMSE'])

        return autoencoder_start_time, autoencoder_end_time, scores, features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df, fv_tdays_dict

    def predict(self, context, model_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inference_start_time = time.time()
        log_start = time.time()

        # Not a list -> 400 Bad Request
        if not isinstance(model_input, list):
            status_code = 400
            item_result = {"status_code": status_code, "error": "Request body must be a list of objects"}
            log_event(None, log_start, time.time(), None, None, model_input, item_result,
                      _api_status_from_code(status_code, "INVALID_FORMAT"),
                      None, time.time(), None, None)
            return [item_result]

        # Empty list -> 400 Bad Request (client error)
        if not model_input:
            status_code = 400
            item_result = {"status_code": status_code, "error": "Empty input list. Provide at least one item."}
            log_event(None, log_start, time.time(), None, None, model_input, item_result,
                      _api_status_from_code(status_code, "EMPTY_INPUT"),
                      None, time.time(), None, None)
            return [item_result]

        results = []
        cpty_threshold, threshold_1, threshold_2 = 0.995, 0.97, 0.96
        callsign = 'fx_anomaly_prediction_' + self.allowed_client_for_this_endpoint

        for _, input_dict in enumerate(model_input):
            item_result = {}
            ID = input_dict.get('UniqueId', None)
            try:
                validated_data = ValidateParams(**input_dict)
                data_dict = validated_data.model_dump()
                expected_client = self.allowed_client_for_this_endpoint
                if expected_client != data_dict['Client']:
                    # Treat client mismatch as 403 Forbidden
                    raise ValueError(f"Invalid client name for this endpoint. Expected: {expected_client}, Got: {data_dict['Client']}")

                data_dict['DealDate'] = datetime.strptime(data_dict['DealDate'], "%Y-%m-%d")
                data_dict['MaturityDate'] = datetime.strptime(data_dict['MaturityDate'], "%Y-%m-%d")

                client_dict = {'Client': data_dict['Client'], 'UniqueId': data_dict['UniqueId']}
                client_dict['Anomaly'] = 'No'
                if 'ContLeg' in data_dict:
                    client_dict['ContLeg'] = data_dict['ContLeg']

                inference_data = {k: v for k, v in data_dict.items() if k not in {'Client', 'UniqueId', 'ContLeg'}}
                inference_data = pd.DataFrame([inference_data], index=[0])

                autoencoder_start_time=None; autoencoder_end_time=None
                autoencoder_start_time, autoencoder_end_time, result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df, fv_tdays_dict =                     self._inference_pipeline(inference_data, client_name=client_dict['Client'].lower())

                if isinstance(result, str):
                    client_dict['Anomaly'] = 'Yes'
                    client_dict['Reason'] = result
                else:
                    df_deviation_all = (features - reconstructed_df).round(3)
                    df_dev_bu_cpty = df_deviation_all.loc[:, df_deviation_all.columns.str.startswith(('Cpty'))]
                    if df_dev_bu_cpty[df_dev_bu_cpty > cpty_threshold].any().sum() >= 1:
                        client_dict['Anomaly'] = 'Yes'
                        client_dict['Reason'] = 'The deal is anomalous.'
                    df_deviation = df_deviation_all.loc[:, ~df_deviation_all.columns.str.startswith(('BUnit','Cpty'))]
                    if df_deviation[df_deviation > threshold_1].any().sum() >= 1:
                        client_dict['Anomaly'] = 'Yes'; client_dict['Reason'] = 'The deal is anomalous.'
                    elif df_deviation[df_deviation > threshold_2].any().sum() > 2:
                        client_dict['Anomaly'] = 'Yes'; client_dict['Reason'] = 'The deal is anomalous.'
                    else:
                        client_dict['Anomaly'] = 'No'; client_dict['Reason'] = ''

                status_code = 200
                item_result = {**client_dict, "status_code": status_code}
                log_event(
                    log_id=ID, log_start_time=log_start, log_end_time=time.time(),
                    model_name=self.allowed_client_for_this_endpoint,
                    model_version=get_model_version(callsign),
                    request_payload=model_input, response_payload=item_result,
                    api_status=_api_status_from_code(status_code),
                    inference_start_time=inference_start_time, inference_end_time=time.time(),
                    autoencoder_start_time=autoencoder_start_time, autoencoder_end_time=autoencoder_end_time
                )

            except ValidationError as e:
                status_code = 422  # schema is syntactically valid JSON but fails validation
                item_result = {"status_code": status_code, "error": f"Input validation error: {e}"}
                log_event(ID, log_start, time.time(), self.allowed_client_for_this_endpoint, get_model_version(callsign),
                          model_input, item_result, _api_status_from_code(status_code),
                          inference_start_time, time.time(), None, None)

            except ValueError as e:
                # Specifically used for client mismatch -> Forbidden
                status_code = 403
                item_result = {"status_code": status_code, "error": str(e)}
                log_event(ID, log_start, time.time(), self.allowed_client_for_this_endpoint, get_model_version(callsign),
                          model_input, item_result, _api_status_from_code(status_code, "CLIENT_MISMATCH"),
                          inference_start_time, time.time(), None, None)

            except KeyError as e:
                status_code = 400
                item_result = {"status_code": status_code, "error": f"Got unexpected or missing key '{e}'"}
                log_event(ID, log_start, time.time(), self.allowed_client_for_this_endpoint, get_model_version(callsign),
                          model_input, item_result, _api_status_from_code(status_code, "KEY_ERROR"),
                          inference_start_time, time.time(), None, None)

            except Exception as e:
                status_code = 500
                item_result = {"status_code": status_code, "error": f"An unexpected error occurred during processing: {str(e)}"}
                log_event(ID, log_start, time.time(), self.allowed_client_for_this_endpoint, get_model_version(callsign),
                          model_input, item_result, _api_status_from_code(status_code),
                          inference_start_time, time.time(), None, None)

            results.append(item_result)
        return results
