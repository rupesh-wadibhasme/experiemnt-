
import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import pickle
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
#import seaborn as sns             
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# %% ---------------------------------------------------------------------
# 2  Static column lists (verbatim from notebook)
# -----------------------------------------------------------------------
master_cols_list = [
    'TranCode', 'Instrument', 'BUnit', 'Cpty',
    'ActivationDate', 'CompletionDate', 'DiaryDate', 'MaturityDate',
    'PrimaryCurr', 'BuySellTypeID', 'AuthorisedStatus', 'CheckedStatus',
    'ComparedStatus', 'PreparedStatus', 'ConfirmedStatus', 'BuyAmount',
    'SellAmount', 'BuyCurr', 'SellCurr', 'BuyBalanceMovement',
    'SellBalanceMovement', 'BuyBalance', 'SellBalance', 'SpotRate',
    'ForwardPoints', 'ForwardRate', 'PortCode', 'DealerID', 'LinkRef1',
    'LinkRef2', 'LinkRef3', 'BuyOurBank', 'BuyOurBankAccountNumber',
    'SellOurBank', 'SellOurBankAccountNumber', 'BuyTheirBank',
    'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber',
]

# Columns dropped after business sign‑off (2025‑06‑11)
drop_cols_master = [
    'CompletionDate', 'LinkRef1', 'LinkRef2', 'LinkRef3', 'BuyOurBank',
    'BuyOurBankAccountNumber', 'SellOurBank', 'SellOurBankAccountNumber',
    'BuyTheirBank', 'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber', 'BuyBalanceMovement', 'SellBalanceMovement',
    'DealerID', 'PortCode', 'CheckedStatus', 'ComparedStatus',
    'ConfirmedStatus', 'BuyBalance', 'SellBalance', 'ForwardRate', 'DiaryDate',
    'PreparedStatus', 'BuySellTypeID', 'AuthorisedStatus',  # dropped 06‑Jun‑25
    'TranCode',                                             # dropped 11‑Jun‑25
]

# -----------------------------------------------------------------------
# 3  Utility functions
# -----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read the Excel file and filter out rows with zero Buy/Sell amounts."""
    df = pd.read_excel(path, header=0)
    df = df[(df.BuyAmount != 0.0) & (df.SellAmount != 0.0)]
    print(f"Loaded {len(df):,} rows (non‑zero Buy/Sell amounts).")
    return df.copy()


def drop_unwanted_data(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """Drop agreed‑upon columns and duplicate rows (keep last)."""
    df = df.drop(columns=cols_to_drop, errors="ignore")
    before = len(df)
    #df = df.drop_duplicates(keep="last")
    print(f"Drop duplicates/cols → size {before:,} → {len(df):,} rows.")
    return df


# ......................................................................
# Derived‑feature engineering
# ......................................................................

def add_derived_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Create Is_weekend_date, TDays, FaceValue; cast BUnit categorical."""
    if 'Factor' in df.columns:
        df = df.drop(columns=['Factor'])

    # Weekend indicator (Mon=1 – Sun=7 → weekend = 1)
    df['Is_weekend_date'] = df.ActivationDate.apply(lambda x: 0 if x.date().isoweekday() < 6 else 1)

    # Tenor in days
    df['TDays'] = (df.MaturityDate - df.ActivationDate).dt.days

    # Ensure BUnit numeric treated as category
    df['BUnit'] = df['BUnit'].astype(str)

    # FaceValue based on Primary currency
    df['FaceValue'] = np.where(
        df.PrimaryCurr == df.BuyCurr, np.abs(df.BuyAmount),
        np.where(df.PrimaryCurr == df.SellCurr, np.abs(df.SellAmount), np.nan)
    )

    # Column groups
    categorical_columns = [
        'Instrument', 'BUnit', 'Cpty',
        'PrimaryCurr', 'BuyCurr', 'SellCurr',
    ]
    numeric_columns = [
        'BuyAmount', 'SellAmount', 'SpotRate',
        'ForwardPoints', 'Is_weekend_date', 'TDays', 'FaceValue',
    ]
    return df, categorical_columns, numeric_columns


# ......................................................................
# Group‑wise clipping and scaling helpers (verbatim logic)
# ......................................................................


def _scale_group(group: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, MinMaxScaler, float, float]:
    scaler = MinMaxScaler()
    group[[column]] = scaler.fit_transform(group[[column]])
    return group, scaler, float(group[column].mean()), float(group[column].std())


def _clip_percentile(group, column, upper_percentile = 0.99, lower_percentile=0.01):
  if len(group[column]) < 2:
    lower = group[column].min()
    upper = group[column].max()
    # percent = 0
    counts = 0
    lower_counts = 0
    upper_counts = 0
    clipped_idx =[]
  else:    
    upper = group[column].quantile(upper_percentile)
    lower = group[column].quantile(lower_percentile)
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper,inclusive='both')].shape[0]
    clipped_idx = group[~group[column].between(lower, upper, inclusive='both')].index
    group[column] = group[column].apply(lambda x: 0 if x>=lower and x<=upper else 1)
  return group, lower, upper, (counts, lower_counts, upper_counts),clipped_idx


def group_points(df, groupby_columns, column, iqr=False, zscore=False, percentile=True, mod_zscore=False,need_scaled_df=True):
  grouped_scalers = {}
  grouped_scaled_dfs = []
  grouped = df.groupby(groupby_columns, sort=False)
  total_counts = 0
  lower_counts = 0
  upper_counts = 0
  clipped_idxs =[]
  for name, group in grouped:
      if percentile:
        group, lower, upper, counts,clipped_idx = _clip_percentile(group, column, upper_percentile=0.99)
      #clipped_idxs.extend(clipped_idx)
      total_counts+=counts[0]
      lower_counts+=counts[1]
      upper_counts+=counts[2]
      if need_scaled_df:
        scaled_group, scaler, mean, sd = _scale_group(group, column)
        grouped_scalers[name] = {'scaler':scaler, 'mean':mean, 'sd':sd, 'lower':lower, 'upper':upper}        
        grouped_scaled_dfs.append(scaled_group)
      else:
        grouped_scaled_dfs.append(group)

  grouped_df = pd.concat(grouped_scaled_dfs)
  return grouped_df, grouped_scalers, (total_counts, lower_counts, upper_counts)#,clipped_idxs
# ......................................................................
# Encoding & final feature frame
# ......................................................................

global_scaler: MinMaxScaler | None = None  # keeps behaviour identical to notebook


def one_hot(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df)
    enc = encoder.transform(df)
    return pd.DataFrame(enc, columns=encoder.get_feature_names_out(df.columns)), encoder


def minmax_scale(df: pd.DataFrame):
    global global_scaler
    if global_scaler is None:
        global_scaler = MinMaxScaler()
    global_scaler.fit(df)
    scaled = global_scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns).fillna(0), global_scaler


def compute_frequency_stats(df: pd.DataFrame, output_path='frequency_stats.pkl'):
    """
    Compute frequency of key categorical combinations for later use in anomaly reasoning.
    Args:
        df (pd.DataFrame): Cleaned training dataframe
        output_path (str): File path to store the stats pickle
    """
    freq_dict = {}

    # Example combinations
    combos_to_count = [        
        ('BUnit', 'Cpty', 'BuyCurr','SellCurr') 
    ]

    for combo in combos_to_count:
        key_counts = Counter(tuple(row) for row in df[list(combo)].values)
        freq_dict.update(key_counts)



def build_feature_frame(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],output_path:str,CLIENT_NAME:str):
    
    # Remove already‑scaled FaceValue/TDays for the generic scaler\
   
    compute_frequency_stats(df, output_path=output_path)

    num_cols_ = [c for c in num_cols if c not in ('FaceValue', 'TDays')]

    cat_df, ohe = one_hot(df[cat_cols])
    num_df, mms = minmax_scale(df[num_cols_])

    
    # append scaled FaceValue & TDays directly (already 0‑1)
    num_df['FaceValue'] = df['FaceValue'].values
    num_df['TDays'] = df['TDays'].values

    # ----------------- Additional logic ----------------------
    print('saving value bounds')

    models_path = output_path          # wherever you save artefacts

    # track index then drop helper column
    cat_df['Index'] = df.index
    features = pd.concat([cat_df, num_df], axis=1)
    features.drop(columns='Index', inplace=True)
    features['FaceValue'].fillna(0, inplace=True)
    return features, ohe, mms


# ......................................................................
# Misc helpers
# ......................................................................

def build_cpty_groups(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for cp in df['Cpty'].unique().tolist():
        out[cp] = {
            'buy': df[df['Cpty'] == cp]['BuyCurr'].unique().tolist(),
            'sell': df[df['Cpty'] == cp]['SellCurr'].unique().tolist(),
        }
    return out


def save_artifacts(CLIENT_NAME,mms: MinMaxScaler, ohe: OneHotEncoder,
                   fv_scalers: Dict, cpty_groups: Dict,
                   tdays_scalers: Dict, *, dst: str):
    os.makedirs(dst, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    fname = f"{CLIENT_NAME}_Expriment_all_scales_NonEmbed_{mms}_{ts}.pkl".replace(' ', '')
    payload = {
        CLIENT_NAME: {
            'mms': mms,
            'ohe': ohe,
            'grouped_scalers': fv_scalers,
            'cpty_group': cpty_groups,
            'tdays_scalers': tdays_scalers,
        }
    }
    with open(os.path.join(dst, fname), 'wb') as fh:
        pickle.dump(payload, fh)
    print(f"Artifacts saved → {fh.name}")


# ......................................................................
# Input‑tensor helper (retained verbatim)
# ......................................................................

def prepare_input_data(features_df: pd.DataFrame,
                        categorical_columns: List[str],
                        numeric_columns: List[str]):
    """Return list‑of‑arrays for model.consume() + per‑input shapes."""
    input_data: List[pd.DataFrame] = []
    input_shapes: List[int] = []

    for category in categorical_columns:
        cols = [c for c in features_df.columns if c.startswith(category)]
        input_shapes.append(len(cols))
        input_data.append(features_df[cols])

    input_shapes.append(len(numeric_columns))
    input_data.append(features_df[numeric_columns])
    return input_data, input_shapes


def find_year_gap_data(df1):
    from dateutil.relativedelta import relativedelta

    result_rows = []

    for i in range(1, len(df1)):
        current = df1.iloc[i]
        previous = df1.iloc[i - 1]
       
        if (current['BUnit'] == previous['BUnit'] and
            current['Cpty'] == previous['Cpty'] and
            current['BuyCurr'] == previous['BuyCurr'] and
            current['SellCurr'] == previous['SellCurr']):

            if current['ActivationDate'] >= previous['ActivationDate'] + relativedelta(months=12):
                result_rows.append(current)

    year_gaps_df = pd.DataFrame(result_rows)
    year_gaps_df['BuyCurr']=df1['BuyCurr']
    year_gaps_df['SellCurr']=df1['SellCurr']
    #year_gaps_df[['BuyCurr', 'SellCurr']] = year_gaps_df['Buy_Sell_Curr_Pair'].str.split('_', expand=True)
    #year_gaps_df.drop('Buy_Sell_Curr_Pair',axis=1, inplace=True)
    return year_gaps_df

#year_gaps_df = find_year_gap_data(df1)
#year_gaps_df



# -----------------------------------------------------------------------
# 4  Main routine
# -----------------------------------------------------------------------
DATA_PATH   = r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\Datasets\training_data\IXOM_Data.xlsx"
MODELS_PATH = r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler_1"
PICKLES_PATH = r"C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler_1"

CLIENT_NAME = "IXOM"
#clnt='lenovo'
if __name__ == "__main__":
    # 0) Load raw data
    raw = load_data(DATA_PATH)

    # 1) Drop unwanted columns / duplicates
    data = drop_unwanted_data(raw, drop_cols_master)
    data['Instrument']=data['Instrument'].str.upper().str.replace(r'\s+', '', regex=True)

    # 2) Feature engineering
    data, cat_cols, num_cols = add_derived_features(data)

    # Create yesr gap dataframe

    year_gaps_df = find_year_gap_data(data)

    
    file_path = PICKLES_PATH+ rf'\{CLIENT_NAME}_year_gap_data.pkl'

    # Save the DataFrame to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(year_gaps_df, f)


    # 3) Group‑wise clipping+scaling (percentile method)
    group_facevalue = ['BUnit', 'Cpty', 'PrimaryCurr']
    group_Instrument = ['Instrument']

    grouped_df, tdays_scalers, _ = group_points(data, group_Instrument, 'TDays')
    grouped_df, fv_scalers, _    = group_points(grouped_df, group_facevalue, 'FaceValue')

    # 4) Save convenience index if needed downstream
    group_clipped_idx = grouped_df.index

    # 5) Counter‑party currency groups
    cpty_groups = build_cpty_groups(data)

    # 6) One‑hot + MinMax scaling → final feature dataframe
    features, ohe, mms = build_feature_frame(grouped_df, cat_cols, num_cols,PICKLES_PATH,CLIENT_NAME)

    # 7) Persist scalers & metadata
    save_artifacts(CLIENT_NAME,mms, ohe, fv_scalers, cpty_groups, tdays_scalers, dst=PICKLES_PATH)

    

    # 8) Optionally build nn‑ready tensors (train/test split not included here)
    #     – replicate notebook logic outside if desired.
    # train_input, input_shapes = prepare_input_data(features, cat_cols, num_cols)
    print("Data‑preparation pipeline complete.")



numeric_columns =['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints', 'Is_weekend_date','TDays','FaceValue']
categorical_columns =['Instrument', 'BUnit', 'Cpty',
                      'PrimaryCurr', 'BuyCurr', 'SellCurr'
                      ]

catg_index_range=[0]
for col in (categorical_columns):
    catg_index_range.append(data[col].nunique())

feature_ohe_cat_index=[]
feature_ohe_all_index=[]
for i in range(1, len(catg_index_range)+1):
    feature_ohe_cat_index.append(sum(catg_index_range[0:i]))
feature_ohe_all_index = feature_ohe_cat_index.copy()
feature_ohe_all_index.append(feature_ohe_all_index[-1]+len(numeric_columns)+1)
del feature_ohe_cat_index

input_dict = dict()
for i in range(len(categorical_columns)+1):
    input_dict[f"input_{i}"] = features.iloc[:,feature_ohe_all_index[i]:feature_ohe_all_index[i+1]]

input_data =[]
for _, value in input_dict.items():
    input_data.append(value)


