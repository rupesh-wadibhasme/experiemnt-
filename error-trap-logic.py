# ---------------------------------------------------------------
# where do we save it?
bounds_path = os.path.join(models_path, f"{client_name}_value_bounds.pkl")

# already existing objects from your notebook:
#   • grouped_scalers   – dict keyed by (BUnit, Cpty, PrimaryCurr)
#   • tdays_scalers     – dict keyed by (TranType,)
#   • data              – full pre-processed training DataFrame
# ---------------------------------------------------------------

value_bounds = {
    "FaceValue": {},     # {(BU, Cpty, PC): [lo_s, hi_s]}
    "TDays"    : {}      # {(TranType,):      [lo_s, hi_s]}
}

# ---------- FaceValue (grouped) --------------------------------
for gkey, scaler_dict in grouped_scalers.items():
    bu, cp, pc = gkey
    subset_raw = data.loc[
        (data.BUnit == bu) &
        (data.Cpty == cp) &
        (data.PrimaryCurr == pc),
        "FaceValue"
    ]

    p01, p99 = np.percentile(subset_raw, [1, 99])

    # convert to 0-1 space with that group's MinMaxScaler
    scaler = scaler_dict["scaler"]          # fitted MinMaxScaler
    lo_s   = scaler.transform([[p01]])[0, 0]
    hi_s   = scaler.transform([[p99]])[0, 0]

    value_bounds["FaceValue"][gkey] = [float(lo_s), float(hi_s)]

# ---------- TDays (per TranType) --------------------------------
for tkey, scaler_dict in tdays_scalers.items():
    (tran_type,) = tkey
    subset_raw   = data.loc[data.TranType == tran_type, "TDays"]

    p01, p99 = np.percentile(subset_raw, [1, 99])

    scaler = scaler_dict["scaler"]
    lo_s   = scaler.transform([[p01]])[0, 0]
    hi_s   = scaler.transform([[p99]])[0, 0]

    value_bounds["TDays"][tkey] = [float(lo_s), float(hi_s)]

# ---------- SAVE ------------------------------------------------
import pickle
pickle.dump(value_bounds, open(bounds_path, "wb"))
print(f"✓ saved scaled 1 %/99 % bounds → {bounds_path}")

--------------------------------------------------------------------------

# one extra load
bounds = pickle.load(open(os.path.join(models_path,
                                       f"{client}_value_bounds.pkl"), "rb"))

lo_fv, hi_fv = bounds["FaceValue"][(bunit, cpty, primary_curr)]
lo_td, hi_td = bounds["TDays"][(tran_type,)]

flag_raw_fv = not(lo_fv <= scaled_facevalue <= hi_fv)
flag_raw_td = not(lo_td <= scaled_tdays     <= hi_td)
