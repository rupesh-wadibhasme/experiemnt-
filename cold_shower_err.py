
def train_and_score(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    train_fn: Callable[[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], object],
    preprocess_fn: Callable[[pd.DataFrame], np.ndarray],
    recon_error_fn: Callable[[object, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Fit model (fixed hyper-params) on train_df and return per-row errors for score_df."""
    train_input,test_input,input_shapes = preprocess_fn(train_df,train='Yes')
    model = train_fn(train_input,test_input,input_shapes)   # y and sample_weight optional
    features,categorical_columns,num_data = preprocess_fn(score_df)
    errs = recon_error_fn(model, features,categorical_columns,num_data)
    errs = np.asarray(errs).reshape(-1)
    return errs
def inference(data, models_path=None,pickles_path=None, model=None,pickled_scaler=None, client_name='IXOM',train='No'):
  # models_path = os.path.join(models_path, client_name+'_R4')
  #Final columns for inference

  col_list = ['Instrument', 'BUnit', 'Cpty','ActivationDate',
       'MaturityDate', 'PrimaryCurr','BuyAmount',
       'SellAmount', 'BuyCurr', 'SellCurr', 'SpotRate', 'ForwardPoints']

  categories_in_numerics = ['BUnit']

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
  
  
  def load_models(model_path,pickles_path,model,pickled_scaler,client_name='IXOM'):
    model = keras.models.load_model(os.path.join(model_path, model))
    load_scalers = pickle.load(open(os.path.join(pickles_path, pickled_scaler), 'rb'))[client_name]
    return model, load_scalers
  
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
  # print('data:', data)
  model, load_scalers = load_models(models_path, pickles_path,model, pickled_scaler, client_name)
  fv_lower = load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['lower']
  fv_upper = load_scalers['grouped_scalers'][(data.iloc[0]['BUnit'], data.iloc[0]['Cpty'], data.iloc[0]['PrimaryCurr'])]['upper']
  tdays_lower = load_scalers['tdays_scalers'][(data.iloc[0]['Instrument'],)]['lower']
  tdays_upper = load_scalers['tdays_scalers'][(data.iloc[0]['Instrument'],)]['upper']
  fv_tdays_dict = {'facevalue_lower':fv_lower,
                 'facevalue_upper':fv_upper,
                 'tdays_lower':tdays_lower,
                 'tdays_upper':tdays_upper,
                 'actual_facevalue':data['FaceValue'].iloc[0],
                 'actual_tdays':data['TDays'].iloc[0]}
  
  print("data['FaceValue'].iloc[0]:",data['FaceValue'].iloc[0])
  data['FaceValue'] = data['FaceValue'].apply(lambda x: 0 if x>=fv_lower and x<=fv_upper else 1)
  data['TDays'] = data['TDays'].apply(lambda x: 0 if x>=tdays_lower and x<=tdays_upper else 1)
  # print("data['FaceValue'],['TDays'] after converting to 0 or 1",data['TDays'],data['FaceValue'])
  numeric_columns = remove_list(numeric_columns, ['TDays'])
  cat_data, ohe = one_hot(data[categorical_columns], load_scalers['ohe'])
  num_data, mms = scale(data[numeric_columns], load_scalers['mms'])
  num_data['FaceValue'] = data['FaceValue'].values
  num_data['TDays'] = data['TDays'].values
  # split data to multi input format as per trained model

  features = pd.concat([cat_data, num_data,], axis=1)
  features['FaceValue'].fillna(0,inplace=True)
  print('-->',features)
  print(type(features))

  if train:
    stratfied_index_train,stratfied_index_test=split_data(data)
    train_df = features.loc[stratfied_index_train]
    test_df = features.loc[stratfied_index_test]
    print(type(train_df))
    print('training df -->',train_df)
    train_input, input_shapes = prepare_input_data(train_df,categorical_columns,num_data)
    test_input, _ = prepare_input_data(test_df,categorical_columns,num_data)

    return train_input,test_input,input_shapes


  # processed_df_list = []
  # for category in categorical_columns:
  #   category_cols = [x for x in cat_data.columns if x.startswith(category)]
  #   category_df = cat_data[category_cols]
  #   processed_df_list.append(category_df)
  # processed_df_list.append(num_data)
  return features,categorical_columns,num_data
  #return prepare_input_data(features,categorical_columns,num_data)

def prepare_input_data(data,categorical_columns,num_data):
  input_data = []
  # input_data_test = []
  input_shapes = []
  print(type(data))
  print(data.head(10))
  for category in categorical_columns:
    category_cols = [x for x in data.columns if x.startswith(category)]
    input_shapes.append(len(category_cols))
    category_df = data[category_cols]
    input_data.append(category_df)
  input_shapes.append(len(num_data.columns))
  input_data.append(data[num_data.columns.tolist()])
  return input_data, input_shapes

def get_predictions(model,input_data):
        reconstructed_features = model.predict(input_data)
        reconstructed_df = pd.DataFrame()
        reconstructed_normalized_df = pd.DataFrame()
        feature_len = len(model.inputs)
        numeric_idx = feature_len-1
        for i in range(feature_len):
                if i != numeric_idx:
                        df1 = pd.DataFrame(np.array([np.where(l == max(l), 1.0, 0.0) for l in reconstructed_features[:][i]]))
                        df2 = pd.DataFrame(np.array(reconstructed_features[:][i]))
                else:
                        df1 = pd.DataFrame(reconstructed_features[:][numeric_idx])
                        df2 = pd.DataFrame(reconstructed_features[:][numeric_idx])
                reconstructed_normalized_df = pd.concat([reconstructed_normalized_df,df1], axis=1)
                reconstructed_df = pd.concat([reconstructed_df,df2], axis=1)

        #reconstructed_normalized_df.columns =  features.columns
        #reconstructed_df.columns = features.columns
        return reconstructed_features,reconstructed_df,reconstructed_normalized_df

def recon_error_fn(model, features,categorical_columns,num_data) -> np.ndarray:
    X = prepare_input_data(features,categorical_columns,num_data)
    reconstructed_features, reconstructed_df, reconstructed_normalized_df =  get_predictions(model,X)
    print('-> prediction complete. computing recon error')
    
    #recon_error=np.mean((X - reconstructed_normalized_df) ** 2, axis=1)  # or your production error metric
    return np.sqrt(np.mean(np.square(features-reconstructed_df), axis=1))

def preprocess_fn(df: pd.DataFrame,models_path=models_path, pickles_path=pickles_path,model=model, pickled_scaler=pickled_scaler, client_name=client_name,train='Yes'):
    # return model-ready X (e.g., column transformer / one-hot with handle_unknown='ignore')
    return inference(df,models_path, pickles_path,model, pickled_scaler, client_name,train)

