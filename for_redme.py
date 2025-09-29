#preprocess.py 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def _get_column_types(df: pd.DataFrame):
        categorical_columns, numeric_columns = [], []
        for feature in df.columns:
            if df[feature].dtype == object:
                categorical_columns.append(feature)
            elif df[feature].dtype in (float, int):
                numeric_columns.append(feature)
        return categorical_columns, numeric_columns

def _one_hot(df: pd.DataFrame, encoder: OneHotEncoder):
    encoded_data = encoder.transform(df)
    return pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))

def _scale(df: pd.DataFrame, scaler: MinMaxScaler):
    scaled_data = scaler.transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns).fillna(0)


def _remove_list(original_list: list, remove_list: list) -> list:
    for i in remove_list:
        if i in original_list:
            original_list.remove(i)
    return original_list

def _face_value(df: pd.DataFrame) -> pd.DataFrame:
    df["FaceValue"] = np.nan
    if df.PrimaryCurr == df.BuyCurr:
        df["FaceValue"] = np.abs(df.BuyAmount)
    elif df.PrimaryCurr == df.SellCurr:
        df["FaceValue"] = np.abs(df.SellAmount)
    return df

def _get_uniques(grouped_scalers):
    unique_BU, unique_cpty, unique_primarycurr = set(), set(), set()
    for i in grouped_scalers:
        BUnit, Cpty, PrimaryCurr = i
        unique_BU.add(BUnit); unique_cpty.add(Cpty); unique_primarycurr.add(PrimaryCurr)

# llm utils 

import os
import json
import requests
from azure.identity import ClientSecretCredential
from utils.log_utils import *

OPENAI_SCOPE = os.getenv("OPENAI_SCOPE")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
print(OPENAI_SCOPE)
print(AOAI_ENDPOINT)
prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.json')

with open(prompt_path, "r") as f:
    system_prompt = json.load(f,strict=False)["system_prompt"]

llm_context = {
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
"DealDate" : "The date at which the contract is started. This is the date which all properties of the contract are agreed and are binding.",
"BUnit" : "A business unit represents a distinct org entity within a client company, such as a division, department or a subsidiary.  Each business unit can operate semi-independently and is responsible for its own deal positions, exposures and financial activities.  It's also a concept that heps create separate tracking and reporting .",
"TDays" : "Number of Transaction Days to complete a deal.",
"FaceValue" : "FaceValue of a deal based on Bunit, Cpty and PrimaryCur",
"Is_weekend_date" : "Confirms whether the deal happened on weekend dates. value '1' means deal happened on weekends.",
}


def _get_llm_input(context_,input_data, df_deviation,fv_tdays_dict):
    llm_input = f'\nData:\n{input_data} \n' + f'\nDeviated features:\n{df_deviation.to_dict()} \n'\
        f"\nfv_tdays_dict : {fv_tdays_dict} \n" + f"\nContext:{context_}"
    return llm_input

def _get_token():
    credential = ClientSecretCredential(
        client_id=AZURE_SP_CLIENT_ID,
        client_secret=AZURE_SP_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID
    )
    access_token = credential.get_token(OPENAI_SCOPE).token
    return access_token

def _get_llm_output(prompt:str, llm_input:str):
    access_token = _get_token()
    assert access_token
    url = AOAI_ENDPOINT
    # headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"bearer {access_token}"
    }
    data = {"messages":[{"role":"system","content":prompt},{"role":"user","content":llm_input}]}
    response = requests.post(url, headers=headers, json=data)
    try:
        result = response.json()
        result = json.loads(result['choices'][0]['message']['content'])['Reason for Anomaly']
    except:
        result = response.text
        outer_json = json.loads(result)
        inner_content = outer_json['choices'][0]['message']['content']
        inner_content_cleaned = inner_content.replace("'", '"')
        inner_json = json.loads(inner_content_cleaned)
        result = inner_json.get("Reason for Anomaly")
    return result
    return unique_BU, unique_cpty, unique_primarycurr
