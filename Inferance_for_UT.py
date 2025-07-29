# %%
import os
import pandas as pd
import numpy as np
import keras
import pickle
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pydantic import BaseModel
from chatbot import *
#from chatbot_config import get_config
#from common import logger, get_env
from typing import List, Tuple
from datetime import datetime

#CONFIG = get_config()
models_list = {
  'rcc': {'model':None, 'scalers':None},
  'lenovo': {'model':None, 'scalers':None}
}

prompt =f"""
You are a helpful assistant specialized in fintech industry.
You are also expert at Machine learning and Deep learning who understand AutoEncoder models, one hot encoding etc.
Your task is to give proper explanation for the Deal being anomaly using the input data.

### Input Data Decsription:
- You will be provided with Deal data and Features decription, i.e 'Data', 'Context' in Json format.
- The 'Data' object contains the BUnit, Cpty, Deviated_Features. Deviated_Features holds the deviation from actual and model prediction values.
- The 'Context' object will have features descriptions which should be helpful to understand features names defition.

### Specific Instructions:
1. Your job is to analyze the "Deviated_Features" values from 'Data'.
2.Always treat the 'BUnit' and 'Cpty' as single 'Business group' and this group owns the Deal data.
3.The explanation should always be Business group centric and other Features with highest deviations values should be clubbed with this group for the explanation.
6.The explanation should starts with " This FX deal with values Business unit '1' , Counterparty 'A' appears anomalous ...", with the actual values of Business unit and Counter party being placed , followed by the features with high deviations.
7.In the Final explanation the feature names should be replaced with meaningful names by referring to 'Context' object. Use Contect definitions for understanding but don't disclose to enduser how its derived.

### **Expected Output format**:
You must return the Final Response in JSON format strictly for each input as follows:
{{
  \"Reason for Anomaly\": \"Reasoning that includes high deviated features.\",
}}

### **Important Considerations** :
    - Always return the output as JSON object as per format mentioned above. No additional text.
    - Always ensure in the explanation the short names should be replaced with Full names that can be derived from 'Context' object.
    - The 'Reason for Anomaly' should not contain the Deviation values. It should only contain Feature names causing anomaly.
    - Always ensure the explanation should be 'Bunit' and 'Cpty' centric and the deviations in other features has to be summarized accordingly.
    - Do not add any additional text after your final decision!
    - Always maintain professional tone while providing final explanation.
    
### Input data: 'Data' and 'Context' object follows here.

You need to go through the 'Data' and follow all the above instructions and provide final response as per **Expected Output format**.
"""

context = {
"Instrument" : "The type of FX transaction - FXSPOT, FXFORWARD or FXSWAP.",
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
"TDays" : "Numer of Transaction Days to complete a deal.",
"FaceValue" : "FaceValue of a deal based on Bunit, Cpty and PrimaryCur",
"Is_weekend_date" : "Confirms whether the deal happened on weekend dates. value '1' means dela happened on weekends.",
}

# Define the Pydantic model for the API parameters
class ValidateParams(BaseModel):
    '''
    Validate the input params.
    '''
    Client: str
    Instrument: str
    BUnit: str
    Cpty: str
    PrimaryCurr: str
    BuyCurr: str
    SellCurr: str
    BuyAmount: float
    SellAmount: float
    SpotRate: float
    ForwardPoints: float
    ActivationDate: datetime
    MaturityDate: datetime
    UniqueId: str
    ContLeg: int = None

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
            #print(df.columns)
            num_array = df.values.astype("float32")
            #print(num_array.shape)
            continue
        arr = onehot_df_to_index(df)
        cat_arrays.append(arr)
        cardinals.append(df.shape[1])
        embed_dims.append(embed_dim_rule(df.shape[1]))
    return cat_arrays, num_array, cardinals, embed_dims


def get_model(model_name:str):
  '''
  Checks all required artifacts are available or not. 
  If not it will load the artifacts like model, scalers.
  Input :
    model_name(client name) in string format
  Returns:
    It returns model, scalers objects.

  '''
  model_path=r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_keras_models"
  scaler_path=r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler_1"
  model = keras.models.load_model(os.path.join(model_path, "IXOM_latest_model_2.keras"))
  scalers = pickle.load(open(os.path.join(scaler_path, "IXOM_Expriment_all_scales_NonEmbed_MinMaxScaler()_2025-07-29-142737.pkl"), 'rb'))['IXOM']
  return model, scalers

def inference(data:pd.DataFrame, year_gap, client_name:str='rcc'):
  '''
  Analyse the datapoints in three stages.
    - Apply Business Logic to check and filter the new BUnit, Cpty, PrimaryCurr.
    - Analyse data points with encoder model and filter the deviated features.
    - Analyse data points by feeding to LLM and provide reasoning.
  Input:
    data - DataFrame object contains the features names like BUnit, Cpty, BuyCurr, SellCurr etc.
  Output:
    It return the DataFrame objects as follws.
      -features: Transformed/Processed features to feed to model.
      - reconstructed_features: These are raw model predictions
      - reconstructed_df: Transformed features after model prediction.
      - reconstructed_normalised_df: Transformed features in one hot representation format.
      - scores: rmse error or anomaly error reason
  '''
  col_list = ['Instrument', 'BUnit', 'Cpty','ActivationDate',
       'MaturityDate', 'PrimaryCurr','BuyAmount',
       'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']

  categories_in_numerics = ['BUnit']

  def get_column_types(df:pd.DataFrame):
    categorical_columns = []
    numeric_columns = []
    for feature in df.columns:
        if df[feature].dtype == object:
          categorical_columns.append(feature)
        elif df[feature].dtype in (float, int) :
          numeric_columns.append(feature)
    return categorical_columns, numeric_columns

  def one_hot(df:pd.DataFrame, encoder=None):
    #print('printing df here',df)
    if encoder is None:
      encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
      encoder.fit(df)

    encoded_data = encoder.transform(df)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
    return encoded_df, encoder

  def scale(df:pd.DataFrame, scaler=None):
    if scaler is None:
      scaler = MinMaxScaler()
      scaler.fit(df)
    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
    return scaled_df, scaler

  def autoencode(input_df_list, model):
    features = pd.concat(input_df_list, axis=1)
    numeric_block_idx = 6

    cat_all, num_all, _, _ = prepare_blocks(input_df_list, numeric_block_idx)
    inputs_all = cat_all + [num_all]
    reconstructed_features = model.predict(inputs_all)

    reconstructed_df = pd.DataFrame()
    reconstructed_normalized_df = pd.DataFrame()
    feature_len = len(input_df_list)
    numeric_idx = feature_len-1
    for i in range(feature_len):
        if i !=numeric_idx:
            df1 = pd.DataFrame(np.array([np.where(l == max(l), 1.0, 0.0) for l in reconstructed_features[:][i]]))
            df2 = pd.DataFrame(np.array(reconstructed_features[:][i]))
        else:
            df1 = pd.DataFrame(reconstructed_features[:][numeric_idx])
            df2 = pd.DataFrame(reconstructed_features[:][numeric_idx])
        reconstructed_normalized_df = pd.concat([reconstructed_normalized_df,df1], axis=1)
        reconstructed_df = pd.concat([reconstructed_df,df2], axis=1)

    reconstructed_normalized_df.columns =  features.columns
    reconstructed_df.columns = features.columns
    
    return features, reconstructed_features, reconstructed_df, reconstructed_normalized_df
  
  def remove_list(original_list:list, remove_list:list) -> list:
    for i in remove_list:
      original_list.remove(i)
    return original_list
  
  def face_value(df:pd.DataFrame) -> pd.DataFrame:
    df["FaceValue"] = np.nan
    if df.PrimaryCurr == df.BuyCurr:
        df["FaceValue"]=np.abs(df.BuyAmount)
    elif df.PrimaryCurr == df.SellCurr:
        df["FaceValue"]=np.abs(df.SellAmount)
    return df
  
  def get_uniques(grouped_scalers):
    unique_BU =set()
    unique_cpty =set()
    unique_primarycurr =set()
    for i in grouped_scalers:
        BUnit, Cpty, PrimaryCurr = i
        unique_BU.add(BUnit)
        unique_cpty.add(Cpty)
        unique_primarycurr.add(PrimaryCurr)
    return unique_BU, unique_cpty, unique_primarycurr
  
  def check_missing_group(unique_BU:list, unique_cpty:list, unique_primarycurr:list, data:pd.DataFrame):
    missed = 0
    #print('starting BY,CPTY check')
    BU = data['BUnit'].values[0]
    CPTY = data['Cpty'].values[0]
    PC = data['PrimaryCurr'].values[0]
    #print('Starting unique test')
    missing_BU = BU not in unique_BU
    missing_Cpty = CPTY not in unique_cpty
    missing_PC = PC not in unique_primarycurr
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
    #print(missed, message)
    return missed, message
  
  def check_currency(data:pd.DataFrame, trained_cptys):
    new = 0
    CP = data['Cpty'].values[0]
    BuyCurr = data['BuyCurr'].values[0]
    SellCurr = data['SellCurr'].values[0]
    missing_BuyCurr = BuyCurr not in trained_cptys[CP]['buy']
    missing_SellCurr = SellCurr not in trained_cptys[CP]['sell']
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
  

  def compare_rows(df1, df2, columns=['BUnit','Cpty','BuyCurr','SellCurr', 'ActivationDate']):
      row = df2.iloc[0]
      mask = (df1[columns] == row[columns].values).all(axis=1)
      return mask.any()

  is_year_gap = compare_rows(year_gap, data)
  if is_year_gap:
    #print('is_year_gap',is_year_gap)
    message =  f"The Currency_pair {data.iloc[0]['BuyCurr'],data.iloc[0]['SellCurr']} happening after more than 1 year making this deal suspicious."
    return message,'', '', '', ''

  #logger.info(f"Preprocessing started...")
  #print(f"Preprocessing started...")
  #start_time = datetime.now()
  data = data[col_list].copy()
  data['Is_weekend_date'] = data.ActivationDate.apply(lambda x: x.date().isoweekday())
  #Convert weekdays to '0' and weekend to '1'
  data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x<6 else 1)
  data['TDays'] = (data.MaturityDate - data.ActivationDate).dt.days
  #end_time = datetime.now()
  #feature_engg = (end_time - start_time).total_seconds()

  #print ('time for feature engg--->',feature_engg)
  #Convert BUnit, TranCode, AuthorisedStatus, DealerID into Categorical.
  data[categories_in_numerics] = data[categories_in_numerics].astype(str)
  #Identify categorical Columns and Numeric columns
  categorical_columns, numeric_columns = get_column_types(data)
  data = pd.concat((data[categorical_columns].astype('category'), data[numeric_columns].astype(float)), axis=1)
  #Facevalue column creation based on Primary Currency
  data = data.apply(face_value, axis=1)
  #Fill missing values
  data.fillna(0, inplace=True)

  # load models an scalers
  #print(f"Loading model instances.")
  model, load_scalers = get_model(client_name)
  #print(load_scalers)
  unique_BU, unique_cpty, unique_primarycurr = get_uniques(load_scalers['grouped_scalers'])

  #print(f"Checking Business Logics.")
  missing, message = check_missing_group(unique_BU, unique_cpty, unique_primarycurr, data)
  if missing:
    return message, '', '', '', ''
  new, message = check_currency(data, load_scalers['cpty_group'])
  if new:
    return message, '', '', '', ''
  
  #start_time = datetime.now()

  # data['FaceValue'] = data.apply(
  #       lambda row: load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['scaler'].transform([[row['FaceValue']]])[0][0], axis=1)
  
  # print('Facevalue-> ',data['FaceValue'])
  # data['TDays'] = data.apply(
  #       lambda row: load_scalers['tdays_scalers'][(row['Instrument'],)]['scaler'].transform([[row['TDays']]])[0][0], axis=1)
  # print('TDays- >',data['TDays'])

  fv_lower = load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['lower']
  fv_upper = load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['upper']
  tdays_lower = load_scalers['tdays_scalers'][(data.iloc[0]['Instrument'],)]['lower']
  tdays_upper = load_scalers['tdays_scalers'][(data.iloc[0]['Instrument'],)]['upper']
  data['FaceValue'] = data['FaceValue'].apply(lambda x: 0 if x>=fv_lower and x<=fv_upper else 1)
  data['TDays'] = data['TDays'].apply(lambda x: 0 if x>=tdays_lower and x<=tdays_upper else 1)

  #print(f"Applying scaling techniques.")
  numeric_columns = remove_list(numeric_columns, ['TDays'])
  #print('Numeric cols->',numeric_columns)
  #print('cat cols->',categorical_columns)
  #print ('loading ohe scaler-->', load_scalers['ohe'])
  #print(data[categorical_columns])

  cat_data, ohe = one_hot(data[categorical_columns], load_scalers['ohe'])
  num_data, mms = scale(data[numeric_columns], load_scalers['mms'])
  
  num_data['FaceValue'] = data['FaceValue'].values
  num_data['TDays'] = data['TDays'].values

  #end_time = datetime.now()
  #Scaling_tranform = (end_time - start_time).total_seconds()
  #print('time required for scaling and transformation-->',Scaling_tranform)


  # split data to multi input format as per trained model
  processed_df_list = []
  for category in categorical_columns:
    category_cols = [x for x in cat_data.columns if x.startswith(category)]
    category_df = cat_data[category_cols]
    processed_df_list.append(category_df)
  processed_df_list.append(num_data)

  #start_time = datetime.now()
  
  #print(processed_df_list)
  features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = autoencode(processed_df_list, model)
  #end_time = datetime.now()
  #autoencode_pred = (end_time - start_time).total_seconds()
  #print('time required for autoencoder-->',autoencode_pred)

  scores = pd.DataFrame(np.sqrt(np.mean(np.square(features-reconstructed_df), axis=1)), columns=['RMSE'])
  return scores, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df
  
def get_llm_output(llm_input:str):
  '''
  Initiate the LLM API and feed the input for explantion.
  Input:
    llm_input - Contain model predictions along with context.
  Output:
    It return output in json format as follows.
    {Reason for anomaly - reason explantion}
  '''
  bot = Chatbot(FISLLM(), KnowledgeBases(), Chat.from_memory())
  bot.chat.add('system', prompt)
  return bot(llm_input)

def get_condition_filters(df_deviation:pd.DataFrame):
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

def get_filtered_data(features:pd.DataFrame, reconstructed_df:pd.DataFrame) -> dict:
  '''
  Creates the final format to feed to LLM.
  Input:
    -features: Raw features after transformation.
    -reconstructed_df: model prediction after transformations.
  Output:
    filtered_data: highly deviated features list along with BUnit and Cpty in json.
  '''
  df_deviation = features-reconstructed_df
  anomaly_count , non_anomaly_count, Deviated_Features = get_condition_filters(df_deviation)
  actual = [col for col in features.columns if (col.startswith('BUnit') or col.startswith('Cpty')) and (features[col] == 1).any()]
  Bunit = actual[0].split('_')
  Cpty = actual[1].split('_')
  filtered_data = {Bunit[0]:Bunit[1], Cpty[0]:Cpty[1], 'Deviated_Features':Deviated_Features}
  return filtered_data

# at top of the file, right after imports

_bounds_cache = {}

def get_value_bounds(client: str, models_path: str):
    """
    Load <client>_value_bounds.pkl exactly once and keep in memory.
    Structure:
       {"FaceValue": [lo_s, hi_s],
        "TDays"    : [lo_s, hi_s]}
    """
    if client in _bounds_cache:
        return _bounds_cache[client]

    path = os.path.join(models_path, f"{client}_value_bounds.pkl")
    with open(path, "rb") as f:
        _bounds_cache[client] = pickle.load(f)
    return _bounds_cache[client]

def anomaly_prediction(data: dict, debug: bool = False) -> dict:
    """
    Perform anomaly detection on an incoming FX deal using autoencoder reconstruction error
    and business rule + statistical + categorical reasoning.

    Args:
        data (dict): Input FX deal with keys including 'Client', 'UniqueId', and transaction features.
        debug (bool): If True, prints internal computation details for debugging.

    Returns:
        dict: {
            'Client': ...,
            'UniqueId': ...,
            'Anomaly': 'Y' / 'N',
            'Reason': explanation string
        }
    """
    
    # Load frequency stats
    models_path = r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler_1'
    year_gap=pickle.load(open(os.path.join(models_path,"IXOM_year_gap_data.pkl"), 'rb'))
    # ------------------------------
    # Constants
    combo_keys = ['BUnit', 'Cpty', 'BuyCurr', 'SellCurr']
    threshold_1 = 0.8
    threshold_2 = 0.9

    # ------------------------------
    # Step 0: Initial Setup
    client_dict = {x: v for x, v in data.items() if x in ['Client', 'UniqueId']}
    client_dict['Anomaly'] = 'N'
    client_dict['Reason'] = 'This FX deal is normal'

    data = {x: v for x, v in data.items() if x not in ['Client', 'UniqueId']}
    client_name = client_dict['Client']

    result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = inference(
        pd.DataFrame(data, index=[0]), year_gap,client_name)

    #print(features)
    print('-----------------------------------------------')
    #print(reconstructed_features)
    if isinstance(result, str):
        # Business rule triggered
        client_dict['Anomaly'] = 'Y'
        client_dict['Reason'] = result
        return client_dict

    # ------------------------------
    # with open(f'{models_path}\\RCC_frequency_stats.pkl', 'rb') as f:
    #     freq_stats = pickle.load(f)

    # ------------------------------
    # Step 2: Extract combination
    # actual_combo = tuple(data[k] for k in combo_keys)
    # sorted_combos = sorted(freq_stats.items(), key=lambda x: x[1])
    # least_freq_combos = set([k for k, _ in sorted_combos[:50]])

    # ------------------------------
    # Step 3: Value bounds for numeric checks
    #bounds = get_value_bounds(client_name, models_path)
    # lo_fv, hi_fv = bounds['FaceValue']
    # lo_td, hi_td = bounds['TDays']

    # ------------------------------
    # Step 4: Deviation DataFrame

    print(reconstructed_df)
    columns = features.columns
    df_deviation = (features - reconstructed_df).abs().round(2)
    df_deviation = pd.DataFrame(df_deviation, columns=columns)

    # ------------------------------
    # Step 5: Start building reasons
    Anomalous = 'N'
    reason_bits = []
    numeric_anomaly = False
    categorical_anomaly = False

    # === Numeric threshold-based anomaly check ===
    # fv_actual = features['FaceValue'].iat[0]
    # td_actual = features['TDays'].iat[0]
    fv_error = df_deviation['FaceValue']
    td_error = df_deviation['TDays']

    print(fv_error)
    print(td_error)
    
    # flag_raw_fv = fv_actual < lo_fv or fv_actual > hi_fv
    # flag_raw_td = td_actual < lo_td or td_actual > hi_td
    flag_err_fv = fv_error > 0.7
    flag_err_td = td_error > 0.7
    #print(type(flag_err_fv[0]),type(flag_err_td[0]))
    if flag_err_fv[0]:
        Anomalous = 'Y'
        reason_bits.append("Deal amount (FaceValue) falls outside the typical range observed for past trades.")
    if flag_err_td[0]:
        Anomalous = 'Y'
        reason_bits.append("Transaction tenor (TDays) differs significantly from comparable historical deals.")  

    # if  flag_err_fv or flag_err_td:
    #     numeric_anomaly = True
    #     Anomalous = 'Y'
    #     if flag_err_fv:
    #         reason_bits.append("Deal amount (FaceValue) falls outside the typical range observed for past trades.")
    #     if flag_err_td :
    #         reason_bits.append("Transaction tenor (TDays) differs significantly from comparable historical deals.")

    # # === Rare combo detection ===
    # if actual_combo in least_freq_combos:
    #     Anomalous = 'Y'
    #     reason_bits.append(
    #         f"The combination of Business Unit '{actual_combo[0]}', Counterparty '{actual_combo[1]}', "
    #         f"Buy Currency '{actual_combo[2]}' and Sell Currency '{actual_combo[3]}' is among the least seen in past data."
    #     )

    # === Decode one-hot categorical values from prediction ===
    def decode_one_hot(df_row, prefix):
        matches = [col for col in df_row.index if col.startswith(f"{prefix}_")]
        if not matches:
            return 'Unknown'
        subrow = df_row[matches]
        return subrow.idxmax().replace(f"{prefix}_", "")

    
    actual_combo = tuple(decode_one_hot(features.loc[0], k) for k in combo_keys)
    predicted_combo = tuple(decode_one_hot(reconstructed_df.loc[0], k) for k in combo_keys)

    # Unpack for clarity
    actual = dict(zip(combo_keys, actual_combo))
    pred = dict(zip(combo_keys, predicted_combo))

    # Track mismatches
    mismatches = {k: (actual[k], pred[k]) for k in combo_keys if actual[k] != pred[k]}
    matches = {k: actual[k] for k in combo_keys if actual[k] == pred[k]}

    # Initialize flags
    categorical_anomaly = False
    #Anomalous = 'N'

    # Case 1: BUnit and Cpty match, but Buy/Sell differ
    if all(k in matches for k in ['BUnit', 'Cpty']) and any(k in mismatches for k in ['BuyCurr', 'SellCurr']):
        categorical_anomaly = True
        Anomalous = 'Y'
        reason_bits.append(
            f"For BUnit '{actual['BUnit']}' and Cpty '{actual['Cpty']}', the currency pair is unexpected. "
            f"Model suggests BuyCurr should be '{pred['BuyCurr']}' and SellCurr should be '{pred['SellCurr']}'."
        )

    # Case 2: Buy/Sell match, but BUnit or Cpty differ
    elif all(k in matches for k in ['BuyCurr', 'SellCurr']) and any(k in mismatches for k in ['BUnit', 'Cpty']):
        categorical_anomaly = True
        Anomalous = 'Y'
        for k in ['BUnit', 'Cpty']:
            if k in mismatches:
                reason_bits.append(
                    f"For currency pair {actual['BuyCurr']}/{actual['SellCurr']}, {k} '{actual[k]}' is unexpected. "
                    f"Model suggests it should be '{pred[k]}'."
                )

    # Case 3: All fields differ or other combinations
    elif mismatches:
        categorical_anomaly = True
        Anomalous = 'Y'
        for k in mismatches:
            reason_bits.append(
                f"{k} '{actual[k]}' is unexpected. Model suggests it should be '{pred[k]}'."
            )


    # Debugging output
    if debug:
        print("\n--- DEBUG INFO ---")
        print(f"Actual Combo: {actual_combo}")
        print(f"Predicted Combo: {predicted_combo}")
        print(f"Is Rare Combo: {actual_combo in least_freq_combos}")
        print(f"Numeric Anomaly: {numeric_anomaly}")
        print(f"Categorical Anomaly: {categorical_anomaly}")
        print(f"Deviation Data:\n{df_deviation}")
        print("------------------\n")

    # ------------------------------
    # Final Result
    client_dict['Anomaly'] = Anomalous
    client_dict['Reason'] = " ".join(reason_bits) if reason_bits else "This FX deal is normal."

    return client_dict

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import datetime
data=pd.read_excel(r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\Datasets\final_test_data\IXOM_Test_Data_25Jul2025.xlsx")
data['Instrument']=data['Instrument'].str.upper().str.replace(r'\s+', '', regex=True)
#print(anomaly_prediction(sample))
data['Instrument']

# %%
output_dict=[]
for i in range(data.shape[0]):
    #try:
        output_dict.append(anomaly_prediction(data.loc[i]))
        # if i>=6:
        #     break       
    # except Exception as e:
    #     print ("issue in",i)
    #     print(e)
    #     #print(data.loc[i])
    #     output_dict.append({})
    #     break
   

# %%
#model_path=r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\backend\pickle_folder"
#model = keras.models.load_model(os.path.join(model_path, "_new_embedding_inferance.keras"))
#scalers = pickle.load(open(os.path.join(model_path, "RCC_updated_all_scales.pkl"), 'rb'))['RCC']
output_dict
count = 0
reason=[]
decision=[]
for record in output_dict:
    #print (record)
    val = record.get('Anomaly', '')
    decision.append(record['Anomaly'])
    reason.append(record['Reason'])
    if isinstance(val, str) and val.strip().lower() == 'y':
        count += 1
count

# %%
#saving results for analysis
data['prediction']=decision
data['Reason']= reason
data.to_excel(r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\Datasets\predicted_results\IXOM_predicted_results_1.xlsx')

# %%
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

# Example Series
y_true = data['Is Anomoly ?']
y_pred = decision

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=['Y', 'N'])
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=['Actual Y', 'Actual N'], columns=['Pred Y', 'Pred N']))

# F1 Score
f1 = f1_score(y_true, y_pred, pos_label='Y',average='binary')
print(f"\nF1 Score (for 'Y' as positive class): {f1:.2f}")


