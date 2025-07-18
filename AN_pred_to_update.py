# at top of the file, right after imports
import pickle, os
_bounds_cache = {}

def get_value_bounds(client: str, models_path: str):
    """
    Load <client>_value_bounds.pkl exactly once and keep in memory.
    Structure:
       {"FaceValue": [lo_s, hi_s],
        "TDays"    : [lo_s, hi_s]}
    """
    if client in _bounds_cache:
        return _bounds_cache[client]

    path = os.path.join(models_path, f"{client}_value_bounds.pkl")
    with open(path, "rb") as f:
        _bounds_cache[client] = pickle.load(f)
    return _bounds_cache[client]





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
    threshold_1, threshold_2 = 0.95, 0.90 
    if flag_raw_fv or flag_raw_td or flag_err_fv or flag_err_td:
        Anomalous = 'Yes'
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
        response = " ".join(reason_bits)  # single business‑friendly sentence
    else:
        Anomalous = 'No'
        response  = (
            f"No material anomalies detected in key deal metrics. "
           
        )
