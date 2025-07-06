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
 
