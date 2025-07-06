
import os
import pandas as pd
import numpy as np
import keras
import pickle
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# %%
import warnings
warnings.filterwarnings('ignore')


def inference(data, models_path, client_name='RCC'):
  # models_path = os.path.join(models_path, client_name+'_R4')
  #Final columns for inference
  col_list = ['TranType', 'BUnit', 'Cpty','ActivationDate',
       'MaturityDate', 'PrimaryCurr','BuyAmount',
       'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']#AuthorisedStatus
 
  categories_in_numerics = ['BUnit']#trancode
 
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
 
  def load_models(model_path, client_name='RCC'):
    model = keras.models.load_model(os.path.join(model_path, "ae-2025-06-18_RCC_f1.keras"))
    load_scalers = pickle.load(open(os.path.join(model_path, "all_scales.pkl"), 'rb'))[client_name]
    return model, load_scalers
 
  def autoencode(input_df_list, model):
    features = pd.concat(input_df_list, axis=1)
    reconstructed_features = model.predict(input_df_list)
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
 
  def check_missing_group(unique_BU, unique_cpty, unique_primarycurr, data):
    missed = 0
    BU = data['BUnit'].values[0]
    CPTY = data['Cpty'].values[0]
    PC = data['PrimaryCurr'].values[0]
    missing_BU = BU not in unique_BU
    missing_Cpty = CPTY not in unique_cpty
    missing_PC = PC not in unique_primarycurr
    missing_BU_Cpty = missing_BU and missing_Cpty
    missing_Cpty_PC = missing_Cpty and missing_PC
    missing_BU_PC = missing_BU and missing_PC
    missing_all = missing_BU and missing_Cpty and missing_PC
    if missing_all:
      missed = 1
      # message =  f"The Business Unit, CounterParty, PrimaryCurrency got a new categorical values as {BU, CPTY, PC}  which was not seen in the data so far making this deal suspicious."
      message = f"This deal appears anomalous because the BusinessUnit, CounterParty, PrimaryCurrency {BU, CPTY, PC} has not previously engaged in any FX contracts."
    elif missing_BU_Cpty:
      missed = 1
      message = f"This deal appears anomalous because the BusinessUnit, CounterParty {BU, CPTY} has not previously engaged in any FX contracts."
      # message =  f"The Business Unit, CounterParty got a new categorical values as {BU, CPTY}  which was not seen in the data so far making this deal suspicious."
    elif missing_Cpty_PC:
      missed = 1
      message = f"This deal appears anomalous because the CounterParty, PrimaryCurrency {CPTY, PC} has not previously engaged in any FX contracts."
      # message =  f"The CounterParty, PrimaryCurrency got a new categorical values as {CPTY, PC}  which was not seen in the data so far making this deal suspicious."
    elif missing_BU_PC:
      missed = 1
      message = f"This deal appears anomalous because the BusinessUnit, PrimaryCurrency {BU, PC} has not previously engaged in any FX contracts."
      # message =  f"The Business Unit, PrimaryCurrency got a new categorical values as {BU, PC}  which was not seen in the data so far making this deal suspicious."
    elif missing_BU:
      missed = 1
      message = f"This deal appears anomalous because the BusinessUnit {BU} has not previously engaged in any FX contracts."
      # message =  f"The Business Unit got a new categorical values as {BU}  which was not seen in the data so far making this deal suspicious."
    elif missing_Cpty:
      missed = 1
      message = f"This deal appears anomalous because the CounterParty {CPTY} has not previously engaged in any FX contracts."
      # message =  f"The CounterParty got a new categorical values as {CPTY}  which was not seen in the data so far making this deal suspicious."
    elif missing_PC:
      missed = 1
      message = f"This deal appears anomalous because the PrimaryCurrency {CPTY} has not previously engaged in any FX contracts."
      # message =  f"The PrimaryCurrency got a new categorical values as {PC} which was not seen in the data so far making this deal suspicious."
    else:
      message = 'No missing data'
    return missed, message
 
  def check_currency(data, trained_cptys):
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
      # message =  f"For CounterParty - {CP} Buy and Sell Currency got a new categorical values as {BuyCurr, SellCurr}  which was not seen in the data so far making this deal suspicious."
    elif missing_BuyCurr:
      new = 1
      message = f"This deal appears anomalous because the CounterParty {CP} with BuyCurrency {BuyCurr} has not previously engaged in any FX contracts."
      # message =  f"For CounterParty - {CP} Buy Currency got a new categorical values as {BuyCurr}  which was not seen in the data so far making this deal suspicious."
    elif missing_SellCurr:
      new = 1
      message = f"This deal appears anomalous because the CounterParty {CP} with SellCurrency {SellCurr} has not previously engaged in any FX contracts."
      # message =  f"For CounterParty - {CP} Sell Currency got a new categorical values as {SellCurr}  which was not seen in the data so far making this deal suspicious."
    else:
      message = "No new data"
    return new, message
 
  def get_zscore(x, mean, sd):
    epsilon=1e-9
    zscore = (x - mean)/(sd+epsilon)
    return zscore
 
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
  # numeric_columns = remove_list(numeric_columns, ['RowNum', 'ContNo'])
 
  # load models an scalers
  model, load_scalers = load_models(models_path, client_name)
  unique_BU, unique_cpty, unique_primarycurr = get_uniques(load_scalers['grouped_scalers'])
  missing, message = check_missing_group(unique_BU, unique_cpty, unique_primarycurr, data)
  if missing:
    return message, '', '', '', '', ''
  new, message = check_currency(data, load_scalers['cpty_group'])
  if new:
    return message, '', '', '', '', ''
 
  data['FaceValue'] = data.apply(
        lambda row: load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['scaler'].transform([[row['FaceValue']]])[0][0], axis=1)
  data['TDays'] = data.apply(
        lambda row: load_scalers['tdays_scalers'][(row['TranType'],)]['scaler'].transform([[row['TDays']]])[0][0], axis=1)
  zs_facevalue = data.apply(lambda row: get_zscore(row['FaceValue'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['mean'], load_scalers['grouped_scalers'][(row['BUnit'], row['Cpty'], row['PrimaryCurr'])]['sd']), axis=1)
  zs_tdays = data.apply(lambda row: get_zscore(row['TDays'], load_scalers['tdays_scalers'][(row['TranType'],)]['mean'], load_scalers['tdays_scalers'][(row['TranType'],)]['sd']), axis=1)
  zscores = pd.DataFrame({'Facevalue':zs_facevalue, 'TDays':zs_tdays})
 
  numeric_columns = remove_list(numeric_columns, ['TDays'])
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
 
  features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = autoencode(processed_df_list, model)
  scores = pd.DataFrame(np.sqrt(np.mean(np.square(features-reconstructed_df), axis=1)), columns=['RMSE'])
 
  return scores, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df, zscores
 


# # LLM

import os
from chatbot import *

# %%
prompt ="""
### General Instructions:
- You are a helpful assistant specialized in fintech industry and also having some understanding of Machine Learning concpets.
- Your task is to give proper explanation for the Deal being anomaly using the 

### Input Data Desription:
- You will be provided with Deal data, i.e 'Data' in Json format, z_scores values of few Features in the 'Z_Score' object which is dictionary object followed by context in CSV format in form of 'context' object.
- The 'Data' object contains the actual Input values and the Difference values wrt the model outputs.
- The 'Z_Score' object will have z_scores values for the numerical features such as 'FaceValue' and 'TDays.
- The 'context' object will have the context on the features and their desricptions which should be helpful in explanantion for anomaly.

### Specific Instructions:
1. Your job is to analyze the "Difference" values of features from 'Data', 'z_score' values from 'Z_Score' objects respectively use the features with higher values for reasoning.
2. For z_scores consider features with values outside the range (-3, 3) for reasoning and ignore features within this range.
3. Always Look into the high deviated features your answer by presenting the 'z_score deviated features' and 'Top deviated Features' and then perform the analysis.
4.Always treat the 'BUnit' and 'Cpty' as single 'Business group' and this group owns the Deal data.
5.The explanation should always be Business group centric and other Features with highest deviations in Difference values or z_score values,
  should be clubbed with this group for the explanantion.
6.The explanation should start with "This FX deal with values Business unit '1', Counterparty 'A' appears anomalous ...", with the actual values of Business unit and Counter party being placed , followed by the features with high deviations.
7.In the Final explanation the feature names should be replaced with meaningful names by refering to 'context' object.
8.Provide your explanation in one or maximum Two sentences. Consider only top deviated features wrt z_scores and Difference values for explanation.



### **Expected Output format**:
You must return the Final Response in JSON format strictly for each input as follows:
{{
  \"Reason for Anomaly\": \"Reasoning that includes high deviated features and z_scores.\",
}}

### **Important Considerations** :
    - Always return the output as JSON object as per format mentioned above.No additional text.
    - Always ensure in the exlanation the short names should be replaced with Fullnames that can be derived from 'context' object.
    - The 'Reason for Anomaly' should not contain the Deviation values. It should only contain Feature names causing anomaly.
    - Always ensure the explanation should be 'Bunit' and 'Cpty' centric and the deviations in other features has to be summarized accrodingly.
    - Do not add any additional text after your final decision!
    - Always maintain professional tone while providing final explanation.

### Input data: 'Data','Z_Score' and 'context' object follows here.
You need to go through the 'Data' and follow all the above instructions and provide final response as per **Expected Output format**.
"""

# %%
context = str({
"TranType" : "The type of FX transaction - FXSPOT, FXFORWARD or FXSWAP.",
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
"TDays" : "Transaction days is a derived feature from difference between Maturity date and Activation Date in terms of days for the deal",
"FaceValue" : "The Derived feature from Buy or Sell amount,which can be interpreted as the Deal amount converted in terms of Primary Currency.",
"Is_weekend_date" : "Confirms whether the deal happened on weekend dates. value '1' means dela happened on weekends.",
})


def get_llm_output(llm_input):
  bot = Chatbot(FISLLM(), KnowledgeBases(), Chat.from_memory())
  bot.chat.add('system', prompt)
  return bot(llm_input)


def get_llm_input(row, context, features, reconstructed_features, zscores):
  dfs = [features, reconstructed_features, features-reconstructed_features]
  llm_data_input = pd.concat([df.iloc[row,:].to_frame() for df in dfs], axis=1)
  llm_data_input.columns = ['Actual', 'Expected', 'Difference']
  llm_input = f'\nData:\n{llm_data_input.to_dict()} \n' +f"\nZ_Scores:\n{zscores.to_dict(orient='records')}\n"+context
  return llm_input

def get_condition_filters(df_deviation):
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

# %%
def get_condition_filters(df_deviation):
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

# %%
def get_filtered_data(features, reconstructed_df, zscores):
  df_deviation = features-reconstructed_df
  anomaly_count , non_anomaly_count, Deviated_Features = get_condition_filters(df_deviation)
  # print('Deviated_Features:', Deviated_Features)
  Z_score = zscores.to_dict(orient='records')[0]
  actual = [col for col in features.columns if (col.startswith('BUnit') or col.startswith('Cpty')) and (features[col] == 1).any()]
  Bunit = actual[0].split('_')
  Cpty = actual[1].split('_')
  filtered_data = {Bunit[0]:Bunit[1], Cpty[0]:Cpty[1], 'Z_score':Z_score, 'Deviated_Features':Deviated_Features}
  return filtered_data

# %% [markdown]
# # Get Responses

# %%
def get_response(row_list, test_data):
  response_list =[]
  anomalous_list = []
  for idx in row_list:
    result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df, zscores = inference(pd.DataFrame(test_data.iloc[idx,:].to_dict(), index=[idx]), model_path, client_name='RCC')
    
    if type(result)==str:
      # test.loc[idx, ['Anomalous', 'Intensity', 'Explanation', 'DeviatedFeatures']] = ['Yes', "High", result, '']
      print(idx, result)
      response_list.append(result)
      anomalous_list.append("Yes")
    else:
      filtered_data = get_filtered_data(features, reconstructed_df, zscores)
      # print("filtered_data:", filtered_data)
      if len(filtered_data['Deviated_Features'])>0:
        llm_input = str(filtered_data)
        outs = get_llm_output(llm_input)['answer']
        Anomalous = 'No'
        try:
          outs_dict = json.loads(outs)
          Explanation = outs_dict['Reason for Anomaly']
          Anomalous = 'Yes'
          # test.loc[idx, ['Anomalous', 'Intensity', 'Explanation', 'DeviatedFeatures']] = [Anomalous, Intensity, Explanation, deviation]
          print(outs_dict)
          anomalous_list.append(Anomalous)
          response_list.append(outs)
        except:
          print('@'*40)
          print(f'Index:{idx}',outs)
          print(f'Index:{idx}',type(outs))
          response_list.append(outs)
          Anomalous = "To be cheked in response"
          anomalous_list.append(Anomalous)
          # print('ERROR',result['RMSE'][0])
      else:
        Anomalous = 'No'
        anomalous_list.append(Anomalous)
        response_list.append('NA')
        print('This FX Deal apeears normal')
  try:
    response_df = pd.DataFrame({"Reason for Anomaly":response_list,"Anomaly":anomalous_list })
  except:
    pass
  return response_list,anomalous_list,response_df


# %% [markdown]
# 0, 6, 13, 17,16

# %%
model_path = r""
test_example = pd.read_excel(r"Demo_FXTestCases.xlsx", header=0, sheet_name='Sheet1')
response_list,anomalous_list,response_df = get_response(test_example)


