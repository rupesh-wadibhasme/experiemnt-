# ---- existing objects you already have ----------
# mms           MinMaxScaler fitted on ALL numerics
# df            full training DataFrame in raw units
# input_data[6] numeric block (scaled 0-1)  (shape N × d)
# recon[-1]     reconstructed numeric block (scaled)

# 1️⃣  VALUE bounds (raw percentile → scaled) -------------------------
value_bounds = {}
for col in critical_num_cols:
    p01, p99 = np.percentile(df[col].values, [1, 99])   # raw
    j        = num_data.columns.get_loc(col)            # column index in scaler
    lo_s     = mms.transform([[p01 if k==j else 0 for k in range(len(num_data.columns))]])[0, j]
    hi_s     = mms.transform([[p99 if k==j else 0 for k in range(len(num_data.columns))]])[0, j]
    value_bounds[col] = [float(lo_s), float(hi_s)]

# 2️⃣  ERROR bounds (99-th pct of abs error) --------------------------
err = np.abs(input_data[6].values - recon[-1])          # N × d
error_bounds = {
    col: float(np.percentile(err[:, num_data.columns.get_loc(col)], 99))
    for col in critical_num_cols
}

# 3️⃣  optional row-RMSE threshold (only those cols) -----------------
col_idx = [num_data.columns.get_loc(c) for c in critical_num_cols]
row_rmse = np.sqrt((err[:, col_idx] ** 2).mean(1))
rmse_thr = float(np.percentile(row_rmse, 99))

# 4️⃣  SAVE TO A SEPARATE PICKLE -------------------------------------
bounds_artifacts = {
    "value_bounds": value_bounds,
    "error_bounds": error_bounds,
    "rmse_thr"    : rmse_thr,
    "critical_cols": critical_num_cols
}

pickle.dump(
    bounds_artifacts,
    open(os.path.join(models_path, f"{client_name}_bounds.pkl"), "wb")
)
print("✓ saved", f"{client_name}_bounds.pkl")



=======================================================================

bounds = pickle.load(open(os.path.join(models_dir, f"{client_key}_bounds.pkl"), "rb"))
val_bounds   = bounds["value_bounds"]
err_bounds   = bounds["error_bounds"]
rmse_thr99   = bounds["rmse_thr"]
crit_cols    = bounds["critical_cols"]


xs   = row_scaled[col]                      # scaled input
xhat = recon_scaled[col]                    # scaled reconstruction
lo_s, hi_s = val_bounds[col]
T           = err_bounds[col]
flag_raw  = xs < lo_s or xs > hi_s
flag_diff = abs(xs - xhat) > T

