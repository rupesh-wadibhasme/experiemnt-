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

  
    print("Data‑preparation pipeline complete.")

d = features.reset_index(drop=True)
d['TDays'].value_counts()


d = features.reset_index(drop=True)
d['FaceValue'].value_counts()

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
