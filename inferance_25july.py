def anomaly_prediction(data: dict, debug: bool = False) -> dict:
    """
    Perform anomaly detection on an incoming FX deal using autoencoder reconstruction error
    and business rule + statistical + categorical reasoning.

    Args:
        data (dict): Input FX deal with keys including 'Client', 'UniqueId', and transaction features.
        debug (bool): If True, prints internal computation details for debugging.

    Returns:
        dict: {
            'Client': ...,
            'UniqueId': ...,
            'Anomaly': 'Y' / 'N',
            'Reason': explanation string
        }
    """
    import pickle
    import pandas as pd
    import numpy as np

    # ------------------------------
    # Constants
    combo_keys = ['BUnit', 'Cpty', 'BuyCurr', 'SellCurr']
    threshold_1 = 0.95
    threshold_2 = 0.90

    # ------------------------------
    # Step 0: Initial Setup
    client_dict = {x: v for x, v in data.items() if x in ['Client', 'UniqueId']}
    client_dict['Anomaly'] = 'N'
    client_dict['Reason'] = 'This FX deal is normal'

    data = {x: v for x, v in data.items() if x not in ['Client', 'UniqueId']}
    client_name = client_dict['Client']

    result, features, reconstructed_features, reconstructed_df, reconstructed_normalized_df = inference(
        pd.DataFrame(data, index=[0]), client_name
    )

    if isinstance(result, str):
        # Business rule triggered
        client_dict['Anomaly'] = 'Y'
        client_dict['Reason'] = result
        return client_dict

    # ------------------------------
    # Step 1: Load frequency stats
    models_path = r'C:\Users\LC5753473\OneDrive - FIS\Documents\FIS_Work\AD_analysis\src\backend\scripts\trained_model_scaler'
    with open(f'{models_path}\\lenovo_frequency_stats.pkl', 'rb') as f:
        freq_stats = pickle.load(f)

    # ------------------------------
    # Step 2: Extract combination
    actual_combo = tuple(data[k] for k in combo_keys)
    sorted_combos = sorted(freq_stats.items(), key=lambda x: x[1])
    least_freq_combos = set([k for k, _ in sorted_combos[:50]])

    # ------------------------------
    # Step 3: Value bounds for numeric checks
    bounds = get_value_bounds(client_name, models_path)
    lo_fv, hi_fv = bounds['FaceValue']
    lo_td, hi_td = bounds['TDays']

    # ------------------------------
    # Step 4: Deviation DataFrame
    columns = features.columns
    df_deviation = (features - reconstructed_df).abs().round(2)
    df_deviation = pd.DataFrame(df_deviation, columns=columns)

    # ------------------------------
    # Step 5: Start building reasons
    Anomalous = 'N'
    reason_bits = []
    numeric_anomaly = False
    categorical_anomaly = False

    # === Numeric threshold-based anomaly check ===
    fv_actual = features['FaceValue'].iat[0]
    td_actual = features['TDays'].iat[0]
    fv_error = df_deviation['FaceValue'].iat[0]
    td_error = df_deviation['TDays'].iat[0]

    flag_raw_fv = fv_actual < lo_fv or fv_actual > hi_fv
    flag_raw_td = td_actual < lo_td or td_actual > hi_td
    flag_err_fv = fv_error > threshold_1
    flag_err_td = td_error > threshold_1

    if flag_raw_fv or flag_err_fv or flag_raw_td or flag_err_td:
        numeric_anomaly = True
        Anomalous = 'Y'
        if flag_raw_fv:
            reason_bits.append("Deal amount (FaceValue) falls outside the typical range observed for past trades.")
        if flag_err_fv and not flag_raw_fv:
            reason_bits.append("Deal amount (FaceValue) is unusual compared with similar historical trades.")
        if flag_raw_td:
            reason_bits.append("Transaction tenor (TDays) is well outside the usual maturity window.")
        if flag_err_td and not flag_raw_td:
            reason_bits.append("Transaction tenor (TDays) differs significantly from comparable historical deals.")

    # === Rare combo detection ===
    if actual_combo in least_freq_combos:
        Anomalous = 'Y'
        reason_bits.append(
            f"The combination of Business Unit '{actual_combo[0]}', Counterparty '{actual_combo[1]}', "
            f"Buy Currency '{actual_combo[2]}' and Sell Currency '{actual_combo[3]}' is among the least seen in past data."
        )

    # === Decode one-hot categorical values from prediction ===
    def decode_one_hot(df_row, prefix):
        matches = [col for col in df_row.index if col.startswith(f"{prefix}_")]
        if not matches:
            return 'Unknown'
        subrow = df_row[matches]
        return subrow.idxmax().replace(f"{prefix}_", "")

    predicted_combo = tuple(decode_one_hot(reconstructed_df.loc[0], k) for k in combo_keys)

    for key, actual, predicted in zip(combo_keys, actual_combo, predicted_combo):
        if actual != predicted:
            categorical_anomaly = True
            Anomalous = 'Y'
            reason_bits.append(
                f"The {key} in this deal ('{actual}') does not align with expected patterns. "
                f"Based on historical data, '{predicted}' is typically more consistent with similar deals."
            )

    # === Final decision based on high dimensional deviation ===
    df_dev_wo_cat = df_deviation.loc[:, ~df_deviation.columns.str.startswith(tuple(combo_keys))]
    if df_dev_wo_cat[df_dev_wo_cat > threshold_1].any().sum() >= 1:
        Anomalous = 'Y'
        reason_bits.append("Anomalous due to high deviation in numeric features.")
    elif df_dev_wo_cat[df_dev_wo_cat > threshold_2].any().sum() > 2:
        Anomalous = 'Y'
        reason_bits.append("Anomalous due to moderate deviation across multiple features.")

    # ------------------------------
    # Debugging output
    if debug:
        print("\n--- DEBUG INFO ---")
        print(f"Actual Combo: {actual_combo}")
        print(f"Predicted Combo: {predicted_combo}")
        print(f"Is Rare Combo: {actual_combo in least_freq_combos}")
        print(f"Numeric Anomaly: {numeric_anomaly}")
        print(f"Categorical Anomaly: {categorical_anomaly}")
        print(f"Deviation Data:\n{df_deviation}")
        print("------------------\n")

    # ------------------------------
    # Final Result
    client_dict['Anomaly'] = Anomalous
    client_dict['Reason'] = " ".join(reason_bits) if reason_bits else "This FX deal is normal."

    return client_dict
