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
    print('result is str->')
    client_dict['Anomaly'] = 'Yes'
    client_dict['Reason'] = result
  
  else:
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
    
    if df_deviation[df_deviation>threshold_1].any().sum()>=1:
      #print('df_deviation>threshold_1')
      #print(df_deviation[df_deviation>threshold_1].dropna(axis=1))
      Anomalous = 'Yes'
      response = f"Anomalous due to High Deviations in features.Deviated features:{df_deviation[df_deviation>threshold_1].dropna(axis=1)}"
    elif df_deviation[df_deviation>threshold_2].any().sum()>2:
      #print('df_deviation>threshold_2')
      #print(df_deviation[df_deviation>threshold_2])
      Anomalous = 'Yes'
      response = f"Anomalous due to High Deviations in features.Deviated features:{df_deviation[df_deviation>threshold_2].dropna(axis=1)}"
    else:
      df_deviation[['FaceValue','TDays']] = 5*df_deviation[['FaceValue','TDays']]
      if df_deviation[df_deviation[['FaceValue','TDays']]>threshold_1].any().sum()>=1:
          print('df_deviation[facevalue/tdays]>threshold_1')
          Anomalous = 'Yes'
          response = f"Anomalous due to High Deviations in features.Deviated features:{df_deviation[df_deviation>threshold_1].dropna(axis=1)}"
      else:
        print('Else else')
        Anomalous = 'No'
        #print('FaceValue:',df_deviation['FaceValue'].values[0])
        #print('TDays:',df_deviation['TDays'].values[0])
        response = f"Non Anomalous as there are no major Deviations in features.\n Deviations: 'FaceValue': {df_deviation['FaceValue'].values[0]},'TDays:':{df_deviation['TDays'].values[0]} " 
    client_dict['Anomaly'] = Anomalous
    client_dict['Reason'] = response

  return client_dict
  
