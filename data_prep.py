# %% [markdown]
# # Data

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kneed

# %%
master_cols_list = ['TranCode', 'TranType', 'BUnit', 'Cpty',
       'ActivationDate', 'CompletionDate', 'DiaryDate', 'MaturityDate',
       'PrimaryCurr', 'BuySellTypeID', 'AuthorisedStatus', 'CheckedStatus',
       'ComparedStatus', 'PreparedStatus', 'ConfirmedStatus', 'BuyAmount',
       'SellAmount', 'BuyCurr', 'SellCurr', 'BuyBalanceMovement',
       'SellBalanceMovement', 'BuyBalance', 'SellBalance', 'SpotRate',
       'ForwardPoints', 'ForwardRate', 'PortCode', 'DealerID', 'LinkRef1',
       'LinkRef2', 'LinkRef3', 'BuyOurBank', 'BuyOurBankAccountNumber',
       'SellOurBank', 'SellOurBankAccountNumber', 'BuyTheirBank',
       'BuyTheirBankAccountNumber', 'SellTheirBank',
       'SellTheirBankAccountNumber',]

# %%
#Drop the columns that are decided already.
drop_cols_master = ['CompletionDate',
'LinkRef1',
'LinkRef2',
'LinkRef3',
'BuyOurBank',
'BuyOurBankAccountNumber',
'SellOurBank',
'SellOurBankAccountNumber',
'BuyTheirBank',
'BuyTheirBankAccountNumber',
'SellTheirBank',
'SellTheirBankAccountNumber',
'BuyBalanceMovement', 
'SellBalanceMovement',
'DealerID',
'PortCode',
'CheckedStatus','ComparedStatus','ConfirmedStatus',
'BuyBalance', 
'SellBalance',
'ForwardRate',
'DiaryDate',
'PreparedStatus', 
'BuySellTypeID',
'AuthorisedStatus', #Can be dropped as per input from Shiva on 06/06/2025, after his discussion with Business Team
'TranCode', #Can be dropped as per input from Shiva on 11/06/2025, after his discussion with Business Team
]


# %%
data_path = r"C:\Users\lc5744086\OneDrive - FIS\AD\TRAC-DEV\RCC\Data files\Copy of RCC_distinct_28052025.xlsx" 

RCC_data_master = pd.read_excel(data_path, header=0)

# %%
# Drop the rows where BuyAmount and SellAmount both Zero

RCC_data = RCC_data_master[(RCC_data_master.BuyAmount != 0.0) & (RCC_data_master.SellAmount != 0.0)]

# %%
rcc_master_idx = RCC_data.index
rcc_master_idx

# %%
data = RCC_data.copy()

# %%
if 'Factor' in data.columns:
    data.drop(labels=['Factor'], inplace=True, axis=1)

# %% [markdown]
# # Drop Duplicates

# %%
def drop_unwanted_data(data,cols_to_drop=drop_cols_master):
    # Drop the rows where TranCode not in (1,9) # For Inference only
    # data = data[data['TranCode'].isin([1,9])]

    # Drop the columns that are discussed and decided based on functional team decision and correlation
    data.drop(columns=cols_to_drop,axis=1,inplace=True)
    # Drop the duplicates
    data = data.drop_duplicates(keep='last')
    print(f'Size of data after dropping duplicate rows and cols: {len(data.index)}')
    return data

data = drop_unwanted_data(data, drop_cols_master)
data_index = data.index

# %%
print(len(data[data.BUnit==2][data.Cpty=='CITI']), ",",
      len(data[data.BUnit==2][data.Cpty=='CITI'][data.SellCurr=='CNY']),",",
      len(data[data.BUnit==2][data.Cpty=='CITI'][data.SellCurr=='CNY'][data.PrimaryCurr=='CNY']),",",
      len(data[data.BUnit==2][data.Cpty=='CITI'][data.SellCurr=='CNY'][data.PrimaryCurr=='USD']),",",
)

# %% [markdown]
# # New Features from Transformations from existing ones

# %%
list_columns = data.columns

# %%
data['Is_weekend_date'] = data.ActivationDate.apply(lambda x: x.date().isoweekday())
#Convert weekdays to '0' and weekend to '1'
data['Is_weekend_date'] = data['Is_weekend_date'].apply(lambda x: 0 if x<6 else 1)
data['TDays'] = (data.MaturityDate - data.ActivationDate).dt.days

#Convert BUnit & TranCode into Categorical.
categories_in_numerics = ['BUnit']
for col in categories_in_numerics:
    data[col] = data[col].astype('str')

# %%
categorical_columns =['TranType', 'BUnit', 'Cpty',
                      'PrimaryCurr', 'BuyCurr', 'SellCurr',
                      ]

numeric_columns =['BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints', 'Is_weekend_date', 'TDays']
data = pd.concat((data[categorical_columns].astype('category'), data[numeric_columns].astype(float)), axis=1)

# %%
#Facevalue column creation based on Primary Currency
import numpy as np
def face_value(df):
    if df.PrimaryCurr == df.BuyCurr:
        df["FaceValue"]=np.abs(df.BuyAmount)
    elif df.PrimaryCurr == df.SellCurr:
        df["FaceValue"]=np.abs(df.SellAmount)
    return df
data["FaceValue"] = np.nan
data = data.apply(face_value, axis=1)
numeric_columns.append('FaceValue')

# %%
swap_mean_tdays = data[data.BUnit=="2"][data.Cpty=='REGN'][data.TranType=='FXSWAP']['TDays'].mean()
swap_std_tdays = data[data.BUnit=="2"][data.Cpty=='REGN'][data.TranType=='FXSWAP']['TDays'].std()
value_3std = 3*swap_std_tdays+swap_mean_tdays

#99percentile
value_99p = data[data.BUnit=="2"][data.Cpty=='REGN'][data.TranType=='FXSWAP']['TDays'].quantile(0.99)
print(f"value_3std : {value_3std:.2f},value_99p : {value_99p:.2f}")

# %% [markdown]
# # Grouping, Clipping and Scaling

# %%
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def scale_group(group, column):
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
 
def clip_mod_zscore(group, column, mod_zscore=3.5):
  if len(group[column])<10:
    lower = group[column].min()
    upper = group[column].max()
    counts = 0
    percent = 0
    lower_counts = 0
    upper_counts = 0
  else:
    median = group[column].median()
    mad = np.median(np.abs(group[column] - median))  
    lower = median - mod_zscore * mad / 0.6745
    upper = median + mod_zscore * mad / 0.6745
    lower_counts = group[group[column]<lower].shape[0]
    upper_counts = group[group[column]>upper].shape[0]
    counts = group[~group[column].between(lower, upper, inclusive='both')].shape[0]
    percent = (counts*100)/group.shape[0]
    group[column] = group[column].clip(lower=lower, upper=upper)
  return group, lower, upper, (counts, lower_counts, upper_counts)

def group_points(df, groupby_columns, column, iqr=False, zscore=False, percentile=False, mod_zscore=False,need_scaled_df=True):
  grouped_scalers = {}
  grouped_scaled_dfs = []
  grouped = df.groupby(groupby_columns, sort=False)
  iqr_bds = {}
  zscore_bds = {}
  total_counts = 0
  lower_counts = 0
  upper_counts = 0
  for name, group in grouped:
      if percentile:
        group, lower, upper, counts = clip_percentile(group, column, upper_percentile=0.99)
        # percentile[name] = {'lb':lower, 'ub':upper}

      total_counts+=counts[0]
      lower_counts+=counts[1]
      upper_counts+=counts[2]
      if need_scaled_df:
        scaled_group, scaler, mean, sd = scale_group(group, column)
        grouped_scalers[name] = {'scaler':scaler, 'mean':mean, 'sd':sd}        
        grouped_scaled_dfs.append(scaled_group)
      else:
        grouped_scaled_dfs.append(group)

  grouped_df = pd.concat(grouped_scaled_dfs)
  return grouped_df, grouped_scalers, (total_counts, lower_counts, upper_counts)
 

# %%
# Using Percentile approach for 'TDays'
group_facevalue = ['BUnit', 'Cpty','PrimaryCurr']
group_trantype = ['TranType']
grouped_df, tdays_scalers, total_counts = group_points(data, group_trantype, 'TDays', percentile=True)#change for other columns
grouped_df, fv_scalers, total_counts = group_points(grouped_df, group_facevalue, 'FaceValue', percentile=True)#change for other columns

# %%
grouped_df.TDays.max(), grouped_df.FaceValue.max()

# %%
group_clipped_idx = grouped_df.index
group_clipped_idx

# %%
# Store Unique Buy/Sell currencies for Counter parties
cpty_groups = {}
for cp in data['Cpty'].unique().tolist():
    buy_unique = data[data['Cpty']==cp]['BuyCurr'].unique().tolist()
    sell_unique = data[data['Cpty']==cp]['SellCurr'].unique().tolist()
    cpty_groups[cp] = {'buy': buy_unique, 'sell': sell_unique}

# %% [markdown]
# # One Hot Encoding and scaling

# %%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
global scaler
scaler = MinMaxScaler()

def one_hot(df, encoder=None):
  if encoder is None:
    encoder = OneHotEncoder(sparse_output=False, 
                            handle_unknown="ignore"
                            )
    encoder.fit(df)
  encoded_data = encoder.transform(df)
  encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
  return encoded_df, encoder

def scale(df):
  global scaler
  if scaler is None:
    scaler = MinMaxScaler()
    print('scaler is None')
  scaler.fit(df)
  scaled_data = scaler.transform(df)
  scaled_df = pd.DataFrame(scaled_data, columns=df.columns).fillna(0)
  return scaled_df, scaler

# %%
# Remove TDays and Facevalue for scaling as they already scaled.Append them later after scaling others.
if 'FaceValue' in numeric_columns:
    numeric_columns.remove('FaceValue')
if 'TDays' in numeric_columns:
    numeric_columns.remove('TDays')
numeric_columns

# %%
cat_cols = [_ for _ in categorical_columns if _ in data.columns]
cat_cols

# %%
cat_data, ohe = one_hot(grouped_df[cat_cols])
num_data, mms = scale(grouped_df[numeric_columns])
cat_data['Index'] = grouped_df.index
num_data['FaceValue'] = grouped_df['FaceValue'].values
num_data['TDays'] = grouped_df['TDays'].values
features = pd.concat([cat_data, num_data,], axis=1)
# features.rename(columns={num_col:f'Scaled {num_col}' for num_col in num_cols}, inplace=True)
features.info()

# %%
features_index = features.Index
features = features.drop(columns='Index')
# feature_idx = [x[-1] for x in features_index]
features['FaceValue'].fillna(0,inplace=True)

# %%
import os
import pickle
import datetime
client_name = 'RCC'
models_path = r"C:\Users\lc5744086\OneDrive - FIS\AD\TRAC-DEV\RCC\Artifacts\models"
pickles_path = r"C:\Users\lc5744086\OneDrive - FIS\AD\TRAC-DEV\RCC\Artifacts\pickle files"

formatted_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')

pickle.dump({client_name: {'mms':mms, 'ohe':ohe, 'grouped_scalers': fv_scalers, 'cpty_group': cpty_groups, 'tdays_scalers': tdays_scalers}}, open(os.path.join(pickles_path, f"all_scales_NonEmbed_{scaler.__str__()}_{formatted_timestamp}.pkl"), 'wb'))
# # Get the current datetime
# # Format the datetime to include date, hours, minutes, and seconds
# formatted_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
# # if not os.path.exists(models_path):
#   # os.makedirs(models_path)
# # pickle.dump({client_name: {'mms':mms, 'ohe':ohe, 'grouped_scalers': fv_scalers, 'cpty_group': cpty_groups, 'tdays_scalers': tdays_scalers}}, open(os.path.join(models_path, f"all_scales_NonEmbed_{formatted_timestamp}.pkl"), 'wb'))


# %% [markdown]
# # Stratify and then train_test_split of data

# %%
data.columns

# %% [markdown]
# ### stratify-2 with sklearn combining 3 cols into one

# %%
train_df.head()

# %%
def prepare_input_data(data):
  input_data = []
  # input_data_test = []
  input_shapes = []
  for category in categorical_columns:
    category_cols = [x for x in data.columns if x.startswith(category)]
    input_shapes.append(len(category_cols))
    category_df = data[category_cols]
    input_data.append(category_df)
  input_shapes.append(len(num_data.columns))
  input_data.append(data[num_data.columns.tolist()])
  return input_data, input_shapes

train_input, input_shapes = prepare_input_data(train_df)
test_input, _ = prepare_input_data(test_df)


