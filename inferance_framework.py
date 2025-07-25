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
  client_dict['Anomaly'] = 'N'
  client_dict['Reason'] = 'This FX deal is normal'
  data = {x:v for x,v in data.items() if x not in ['Client', 'UniqueId']}
  client_name=client_dict['Client']
  result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = inference(pd.DataFrame(data, index=[0]), client_name)
  #return result
  
  models_path=r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler'
  if type(result)==str:
    print('Test from encoder')
    client_dict['Anomaly'] = 'Y'
    client_dict['Reason'] = result
  
  else:
    # Adding logic of frequency check +++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Step 1: Load frequency stats (call this once globally or pass into inference)
    with open(r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler\lenovo_frequency_stats.pkl', 'rb') as f:
        freq_stats = pickle.load(f)
    

    # Step 2: Extract the actual combination from input
    combo_keys = ['BUnit', 'Cpty', 'BuyCurr', 'SellCurr']
    actual_combo = tuple(data[k] for k in combo_keys) #tuple(pd.DataFrame(data).loc[0, k] for k in combo_keys)

    # Step 3: Identify least frequent combinations from training (bottom 2)
    sorted_combos = sorted(freq_stats.items(), key=lambda x: x[1])
    least_freq_combos = set([k for k, _ in sorted_combos[:50]])
    reason_bits = []
    #print('Actual combo->', actual_combo)
    #print(least_freq_combos)
    # Step 4: Check if current combo is among least frequency
        

    # ends here -------------------------------------------------------

    bounds= get_value_bounds(client_name,models_path)
    lo_fv, hi_fv = bounds['FaceValue']
    lo_td, hi_td = bounds['TDays']
    threshold_1,threshold_2 = 0.95, 0.9
    # filtered_data = get_filtered_data(features, reconstructed_df)
    # if len(filtered_data['Deviated_Features'])>0:
    #   print(f"Feeding the encoder model prediction to LLM for explanation.")
    #   llm_input = f'\nData:\n{filtered_data} \n' +f"\nContext:\n{context}\n"
    #   outs = get_llm_output(llm_input)['answer']
    #   client_dict['Anomaly'] = 'Yes'
    #   try:
    #     outs_dict = json.loads(outs)
    #     client_dict['Reason'] = outs_dict['Reason for Anomaly']
    #   except:
    #     client_dict['Reason'] = outs
    # else:
    #     client_dict['Reason'] = "This FX deal looks normal."

    # Could be anomaly or non anomaly to be decided by AutoEncoder
    columns = features.columns
    Anomalous = 'Not yet'
    df_deviation = np.abs(features-reconstructed_df)
    df_deviation = df_deviation.round(2)
    df_deviation = pd.DataFrame(df_deviation,columns=columns)
    #Check if any rare combination of Bunit and Cpty
    df_dev_bu_cpty = df_deviation.loc[:,df_deviation.columns.str.startswith(('Cpty'))]
    is_cpty_dev = df_dev_bu_cpty.where(df_dev_bu_cpty>threshold_1).dropna(axis=1)
    # if df_dev_bu_cpty[df_dev_bu_cpty>threshold_1].any().sum()>=1:
    #   print('df_dev_bu_cpty>threshold_1')
    #   Anomalous = 'Yes'
    #   response = f"Anomalous due to High Deviations in features.Deviated features:{df_dev_bu_cpty[df_dev_bu_cpty>threshold_1].dropna(axis=1)}"
    #   response_list.append((idx,Anomalous,response))
    #   continue
    df_deviation = df_deviation.loc[:, ~df_deviation.columns.str.startswith(('BUnit', 'Cpty'))]    
    
    if actual_combo in least_freq_combos:
        print('----> Got rare combo')
        Anomalous = 'Y'
        reason_bits.append(
            f"The combination of Business Unit '{actual_combo[0]}', Counterparty '{actual_combo[1]}', "
            f"Buy Currency '{actual_combo[2]}' and Sell Currency '{actual_combo[3]}' is one of the least frequently seen in past transactions."
        )

         # Step 4: Reverse one-hot for each group to find most probable prediction
        def decode_one_hot(df_row, prefix):
            matches = [col for col in df_row.index if col.startswith(f"{prefix}_")]
            if not matches:
                return 'Unknown'
            subrow = df_row[matches]
            return subrow.idxmax().replace(f"{prefix}_", "")

        # Step 5: Decode reconstructed one-hot columns
        predicted_combo = tuple(decode_one_hot(reconstructed_df.loc[0], k) for k in combo_keys)

        # Step 6: Identify differences
        differing_parts = [
            f" predicted '{predicted}'"
            for key, actual, predicted in zip(combo_keys, actual_combo, predicted_combo)
            if actual != predicted
        ]

        if differing_parts:
            diffs_str = ', '.join(differing_parts)
            reason_bits.append(
                f"Based on similar historical data, it should have been : {diffs_str}."
            )
        #response = " ".join(reason_bits)

    if df_deviation[df_deviation>threshold_1].any().sum()>=1:
      #print('df_deviation>threshold_1')
      #print(df_deviation[df_deviation>threshold_1].dropna(axis=1))
      Anomalous = 'Y'
      reason_bits.append(f"Anomalous due to High Deviations in features.Deviated features:{df_deviation[df_deviation>threshold_1].dropna(axis=1)}")
    elif df_deviation[df_deviation>threshold_2].any().sum()>2:
      #print('df_deviation>threshold_2')
      #print(df_deviation[df_deviation>threshold_2])
      Anomalous = 'Y'
      reason_bits.append(f"Anomalous due to High Deviations in features.Deviated features:{df_deviation[df_deviation>threshold_2].dropna(axis=1)}")
    # ---------- ❷‑B new FaceValue / TDays logic --------------------
    else:
        fv_actual = features['FaceValue'].iat[0]
        td_actual = features['TDays'].iat[0]
        fv_error  = df_deviation['FaceValue'].iat[0]
        td_error  = df_deviation['TDays'].iat[0]

        # ① value rule  (outside 1‑99 % band in scaled space)
        flag_raw_fv = fv_actual < lo_fv or fv_actual > hi_fv
        flag_raw_td = td_actual < lo_td or td_actual > hi_td

        # ② error rule  (> threshold_1 after scaling)
        flag_err_fv = fv_error > threshold_1
        flag_err_td = td_error > threshold_1
        #threshold_1, threshold_2 = 0.95, 0.90 
        if flag_raw_fv or flag_raw_td or flag_err_fv or flag_err_td:
            Anomalous = 'Y'
            reason_bits = []
            if flag_raw_fv:
                reason_bits.append(
                    "Deal amount (FaceValue) falls outside the typical range observed for past trades."
                )
            if flag_err_fv and not flag_raw_fv:
                reason_bits.append(
                    "Deal amount (FaceValue) is unusual compared with similar historical trades."
                )
            if flag_raw_td:
                reason_bits.append(
                    "Transaction tenor (TDays) is well outside the usual maturity window."
                )
            if flag_err_td and not flag_raw_td:
                reason_bits.append(
                    "Transaction tenor (TDays) differs significantly from comparable historical deals."
                )
            # response = " ".join(reason_bits)  # single business‑friendly sentence
        else:
            Anomalous = 'N'
            reason_bits  = ["This FX deal is normal."]
    response= " ".join(reason_bits) 
    client_dict['Anomaly'] = Anomalous
    client_dict['Reason'] = response

  return client_dict
  
