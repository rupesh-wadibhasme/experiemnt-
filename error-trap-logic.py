# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
client_name = "RCC"               # or "LENOVO" …
models_path = "./models"          # wherever you save artefacts
bounds_path = os.path.join(models_path,
                           f"{client_name}_value_bounds.pkl")

# -----------------------------------------------------------
# COLLECT BOUNDS  (no MinMaxScaler needed)
# -----------------------------------------------------------
value_bounds = {}

for col in ["FaceValue", "TDays"]:
    # 1️⃣  raw training percentiles
    p01, p99 = np.percentile(data[col].values, [1, 99])

    # 2️⃣  raw min & max  (same ones the scaler would have stored)
    col_min = data[col].min()
    col_max = data[col].max()

    # guard against divide-by-zero in degenerate case
    rng = max(col_max - col_min, 1e-12)

    # 3️⃣  convert to 0-1 scaled space
    lo_s = (p01 - col_min) / rng
    hi_s = (p99 - col_min) / rng

    value_bounds[col] = [float(lo_s), float(hi_s)]

# -----------------------------------------------------------
# SAVE
# -----------------------------------------------------------
import pickle, os
os.makedirs(models_path, exist_ok=True)
pickle.dump(value_bounds, open(bounds_path, "wb"))
print("✓ saved scaled 1 % / 99 % bounds →", bounds_path)

#----------------------------------------------------------------

bounds = pickle.load(open(bounds_path, "rb"))
lo_fv, hi_fv = bounds["FaceValue"]
flag_raw_fv  = not (lo_fv <= scaled_facevalue <= hi_fv)



