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
"FX_Instrument_Type" : "The type of FX transaction - FXSPOT, FXFORWARD or FXSWAP.",
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
            print(df.columns)
            num_array = df.values.astype("float32")
            print(num_array.shape)
            continue
        arr = onehot_df_to_index(df)
        cat_arrays.append(arr)
        cardinals.append(df.shape[1])
        embed_dims.append(embed_dim_rule(df.shape[1]))
    return cat_arrays, num_array, cardinals, embed_dims



# def load_models():
#   '''
#   Looks for AD_CONN_STRING, AD_CONTAINER, AD_Scalers, AD_MODEL env variables.
#   Connects to Blob storage and read the model and pickle files from the model_artifacts folder.
#   Return:
#   Returns the model, scalers objects.
#   '''
#   container_client = BlobServiceClient.from_connection_string(CONFIG['AD_BLOB_URL']) \
#       .get_container_client(CONFIG["AD_CONTAINER"])
#   scaler_client = container_client.get_blob_client(CONFIG["AD_Scalers"])
#   model_client = container_client.get_blob_client(CONFIG["AD_MODEL"])
#   blob_data = model_client.download_blob().readall()
#   # Save the blob data to a temporary file
#   with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_file:
#     temp_file.write(blob_data)
#     temp_file_path = temp_file.name
#   model = keras.models.load_model(temp_file_path)
#   os.remove(temp_file_path)

#   stream = BytesIO()
#   scaler_client.download_blob().download_to_stream(stream)
#   stream.seek(0)# Reset the stream position to the beginning
#   # Load the pickle file from the stream
#   scalers = pickle.load(stream)
#   return model, scalers

# def load_models(model_path=r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\backend\pickle_folder", client_name='RCC'):
#   model = keras.models.load_model(os.path.join(model_path, "RCC_embedding_inferance.keras"))
#   scalers = pickle.load(open(os.path.join(model_path, "all_scales_5.pkl"), 'rb'))[client_name]
#   return model, scalers

def get_model(model_name:str):
  '''
  Checks all required artifacts are available or not. 
  If not it will load the artifacts like model, scalers.
  Input :
    model_name(client name) in string format
  Returns:
    It returns model, scalers objects.

  '''
  model_path=r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\backend\pickle_folder"
  model = keras.models.load_model(os.path.join(model_path, "RCC_embedding_inferance.keras"))
  scalers = pickle.load(open(os.path.join(model_path, "all_scales_5.pkl"), 'rb'))['rcc']
  #global models_list
  #if models_list[model_name]['model'] is None:
    #models_list[model_name]['model'], models_list[model_name]['scalers'] = load_models()
  #return models_list[model_name]['model'], models_list[model_name]['scalers'][model_name]
  return model, scalers

def inference(data:pd.DataFrame, client_name:str='rcc'):
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
  col_list = ['FX_Instrument_Type', 'BUnit', 'Cpty','ActivationDate',
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
    print('starting BY,CPTY check')
    BU = data['BUnit'].values[0]
    CPTY = data['Cpty'].values[0]
    PC = data['PrimaryCurr'].values[0]
    print('Starting unique test')
    missing_BU = BU not in unique_BU
    missing_Cpty = CPTY not in unique_cpty
    missing_PC = PC not in unique_primarycurr
    missing_BU_Cpty = missing_BU and missing_Cpty
    missing_Cpty_PC = missing_Cpty and missing_PC
    missing_BU_PC = missing_BU and missing_PC
    missing_all = missing_BU and missing_Cpty and missing_PC
    print('starting if cond',missing_all)
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
    print(missed, message)
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
  
  #logger.info(f"Preprocessing started...")
  print(f"Preprocessing started...")
  data = data[col_list].copy()
  data['Is_weekend_date'] = data.ActivationDate.apply(lambda x: x.date().isoweekday())
  #Convert weekdays to '0' and weekend to '1'
  data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x<6 else 1)
  data['TDays'] = (data.MaturityDate - data.ActivationDate).dt.days

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
  print(f"Loading model instances.")
  model, load_scalers = get_model(client_name)
  #print(load_scalers)
  unique_BU, unique_cpty, unique_primarycurr = get_uniques(load_scalers['grouped_scalers'])

  print(f"Checking Business Logics.")
  missing, message = check_missing_group(unique_BU, unique_cpty, unique_primarycurr, data)
  if missing:
    return message, '', '', '', ''
  new, message = check_currency(data, load_scalers['cpty_group'])
  if new:
    return message, '', '', '', ''

  data['FaceValue'] = data.apply(
        lambda row: load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])].transform([[row['FaceValue']]])[0][0], axis=1)
  data['TDays'] = data.apply(
        lambda row: load_scalers['tdays_scalers'][(row['FX_Instrument_Type'],)].transform([[row['TDays']]])[0][0], axis=1)
  print(f"Applying scaling techniques.")
  numeric_columns = remove_list(numeric_columns, ['TDays'])
  #print('Numeric cols->',numeric_columns)
  #print('cat cols->',categorical_columns)
  cat_data, ohe = one_hot(data[categorical_columns], load_scalers['ohe'])
  num_data, mms = scale(data[numeric_columns], load_scalers['mms'])
  num_data['FaceValue'] = data['FaceValue'].values
  num_data['TDays'] = data['TDays'].values
  # split data to multi input format as per trained model
  processed_df_list = []
  for category in categorical_columns:
    category_cols = [x for x in cat_data.columns if x.startswith(category)]
    category_df = cat_data[category_cols]
    processed_df_list.append(category_df)
  processed_df_list.append(num_data)
  print(f"Feeding data to encoder model.",processed_df_list)
  features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = autoencode(processed_df_list, model)
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

def anomaly_prediction(data:dict):
  '''
  Process the input data points and send to repective method.
  Input:
    -data: dictionary with fx deal data points.
  Output:
    Return the anomaly status.
    Anomaly:Yes/No
    Reason : reson for anomaly
    Client: client name
    UniqueId: unique contract id.
  '''
  #data = ValidateParams(**data)
  #data = dict(data)
  client_dict = {x:v for x,v in data.items() if x in ['Client', 'UniqueId']}
  client_dict['Anomaly'] = 'No'
  data = {x:v for x,v in data.items() if x not in ['Client', 'UniqueId']}
  result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = inference(pd.DataFrame(data, index=[0]), client_name=client_dict['Client'])
  #return result

  if type(result)==str:
    client_dict['Reason'] = result
  else:
    filtered_data = get_filtered_data(features, reconstructed_df)
    if len(filtered_data['Deviated_Features'])>0:
      print(f"Feeding the encoder model prediction to LLM for explanation.")
      llm_input = f'\nData:\n{filtered_data} \n' +f"\nContext:\n{context}\n"
      outs = get_llm_output(llm_input)['answer']
      client_dict['Anomaly'] = 'Yes'
      try:
        outs_dict = json.loads(outs)
        client_dict['Reason'] = outs_dict['Reason for Anomaly']
      except:
        client_dict['Reason'] = outs
    else:
        client_dict['Reason'] = "This FX deal looks normal."
  return client_dict
