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

class _RestrictedUnpickler(pickle.Unpickler):
    """Only allow harmless built-ins; block everything else (prevents RCE)."""
    _ALLOWED = {"dict", "list", "tuple", "set", "str", "int", "float", "bool", "bytes"}

    def find_class(self, module, name):
        if module == "builtins" and name in self._ALLOWED:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"Forbidden global during unpickling: {module}.{name}")

    # Disallow external persistent references
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

        # Load the dictionary of scalers and encoders
        #with open(context.artifacts["scalers_path"], 'rb') as f:
            #self.load_scalers = pickle.load(f)[self.allowed_client_for_this_endpoint]
        # SAFE: restricted unpickle of dict/list-of-dicts only
        
        # ------------- safe loader ----------
        with open(context.artifacts["scalers_path"], "rb") as f:
            scalers_all = _safe_pickle_load(f)

        self.load_scalers = scalers_all[self.allowed_client_for_this_endpoint]
        # ------------- ends here -------------
        
        # Pre-extract unique values from scalers for faster lookup
        self.unique_BU, self.unique_cpty, self.unique_primarycurr = self._get_uniques(self.load_scalers['grouped_scalers'])

        # Initialize the Pydantic validator
        # self.ValidateParams = ValidateParams
        # logger.info("Model and scalers loaded successfully.")

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
        missing_BU = BU not in self.unique_BU
        missing_Cpty = CPTY not in self.unique_cpty
        missing_PC = PC not in self.unique_primarycurr
        missing_BU_Cpty = missing_BU and missing_Cpty
        missing_Cpty_PC = missing_Cpty and missing_PC
        missing_BU_PC = missing_BU and missing_PC
        missing_all = missing_BU and missing_Cpty and missing_PC
        if missing_all:
            missed = 1
            message = f"This deal appears anomalous because the BusinessUnit, CounterParty, PrimaryCurrency {BU, CPTY, PC} has not previously engaged in any FX contracts."
        elif missing_BU_Cpty:
            missed = 1
            message = f"This deal appears anomalous because the BusinessUnit, CounterParty {BU, CPTY} has not previously engaged in any FX contracts."
        elif missing_Cpty_PC:
            missed = 1
            message = f"This deal appears anomalous because the CounterParty, PrimaryCurrency {CPTY, PC} has not previously engaged in any FX contracts."
        elif missing_BU_PC:
            missed = 1
            message = f"This deal appears anomalous because the BusinessUnit, PrimaryCurrency {BU, PC} has not previously engaged in any FX contracts."
        elif missing_BU:
            missed = 1
            message = f"This deal appears anomalous because the BusinessUnit {BU} has not previously engaged in any FX contracts."
        elif missing_Cpty:
            missed = 1
            message = f"This deal appears anomalous because the CounterParty {CPTY} has not previously engaged in any FX contracts."
        elif missing_PC:
            missed = 1
            message = f"This deal appears anomalous because the PrimaryCurrency {CPTY} has not previously engaged in any FX contracts."
        else:
            message = 'No missing data'
        return missed, message

     def _check_currency(self, data: pd.DataFrame):
        new = 0
        CP = data['Cpty'].values[0]
        BuyCurr = data['BuyCurr'].values[0]
        SellCurr = data['SellCurr'].values[0]
        
        # Access self.load_scalers['cpty_group']
        if CP not in self.load_scalers['cpty_group']:
            # Handle case where Cpty itself is new
            new = 1
            message = f"This deal appears anomalous because the CounterParty {CP} has not previously engaged in any FX contracts."
            return new, message

        trained_cptys_for_cp = self.load_scalers['cpty_group'][CP]
        missing_BuyCurr = BuyCurr not in trained_cptys_for_cp.get('buy', {})
        missing_SellCurr = SellCurr not in trained_cptys_for_cp.get('sell', {})
        
        missing_BuySell = missing_BuyCurr and missing_SellCurr
        if missing_BuySell:
            new = 1
            message = f"This deal appears anomalous because the CounterParty {CP} with SellCurrency {SellCurr} and BuyCurrency {BuyCurr} has not previously engaged in any FX contracts."
        elif missing_BuyCurr:
            new = 1
            message = f"This deal appears anomalous because the CounterParty {CP} with BuyCurrency {BuyCurr} has not previously engaged in any FX contracts."
        elif missing_SellCurr:
            new = 1
            message = f"This deal appears anomalous because the CounterParty {CP} with SellCurrency {SellCurr} has not previously engaged in any FX contracts."
        else:
            message = "No new data"
        return new, message
    
     def _inference_pipeline(self, data: pd.DataFrame, client_name: str = 'rcc'):
        col_list = ['FX_Instrument_Type', 'BUnit', 'Cpty','DealDate',
                    'MaturityDate', 'PrimaryCurr','BuyAmount',
                    'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']
        categories_in_numerics = ['BUnit'] # Your code casts BUnit to str then to category
        
        # logger.info(f"Preprocessing started for client: {client_name}.")
        data = data[col_list].copy()
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
            return message, '', '', '', ''
        new, message = self._check_currency(data)
        if new:
            return message, '', '', '', ''
        
        # Apply grouped scalers
        # Ensure the column used for lookup (e.g., 'BUnit', 'Cpty', 'PrimaryCurr') are in the correct format (e.g., string)
        data['FaceValue'] = data.apply(
            lambda row: self.load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])].transform(
                [[row['FaceValue']]]
            )[0][0], axis=1)
        data['TDays'] = data.apply(
            lambda row: self.load_scalers['tdays_scalers'][(row['FX_Instrument_Type'])].transform(
                [[row['TDays']]]
            )[0][0], axis=1)
        
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
        features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df = self._autoencode(processed_df_list)
        scores = pd.DataFrame(np.sqrt(np.mean(np.square(features - reconstructed_df), axis=1)), columns=['RMSE'])
        
        return scores, features, reconstructed_features_raw, reconstructed_df, reconstructed_normalized_df
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
        if not isinstance(model_input, list):
            item_result = {
                    "status_code": 500,
                    "error": "Input format is not List of dictionaries",
                }
            return [item_result]
        
        if not model_input:
            # If an empty list is provided, return an empty list of results.
            # The HTTP status code will typically be 200 OK.
            item_result = {
                    "status_code": 200,
                    "message": "Send valid input.",
                }
            return [item_result]

        results = []

        for i, input_dict in enumerate(model_input):
            item_result = {} # Store result for the current item
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
                result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = \
                    self._inference_pipeline(inference_data, client_name=client_dict['Client'].lower())

                if isinstance(result, str): # Anomaly detected by business rules
                    client_dict['Anomaly'] = 'Yes'
                    client_dict['Reason'] = result
                else: # Proceed with autoencoder anomaly detection
                    filtered_data = self._get_filtered_data(features, reconstructed_df)
                    if len(filtered_data.get('Deviated_Features', [])) > 0:
                        client_dict['Anomaly'] = 'Yes'
                        client_dict['Reason'] = 'The deal is anomalous.'
                    else:
                        client_dict['Reason'] = "This FX deal looks normal."

                # Add success status to the item result
                item_result = {**client_dict, "status_code": 200} # Merging client_dict and status

            except ValidationError as e:
                item_result = {
                    "status_code": 400,
                    "error": f"Input validation error: {e.errors()}",
                }
            except KeyError as e:
                item_result = {
                    "status_code": 400,
                    "error": f"Got unexpected key '{e}'",
                }
            except Exception as e:
                # For unexpected errors during processing of an item, capture the error
                # but don't re-raise immediately, allow other items to process.
                item_result = {
                    "status_code": 500,
                    "error": f"An unexpected error occurred during processing: {str(e)}",
                }

            results.append(item_result)

        return results
