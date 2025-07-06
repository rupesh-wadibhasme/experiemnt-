"""
run_batch.py
------------
Reads a test Excel file, feeds each row into inference_one_row(), and
writes predictions.xlsx  (or .csv) with the two new columns.
"""
import pandas as pd
from fx_anomaly_inference import inference_one_row, load_models

# ── CONFIG ───────────────────────────────────────────────────────────
MODELS_DIR   = "./models"                # where the .keras & .pkl live
CLIENT_NAME  = "RCC"                     # key in scaler pickle
TEST_XLSX    = "Demo_FXTestCases.xlsx"
OUTPUT_FILE  = "predictions.xlsx"        # keep Excel; change to .csv if needed

# ── load artefacts once ─────────────────────────────────────────────
_model, _ = load_models(MODELS_DIR, CLIENT_NAME)     # just to warm up Keras

# ── read test sheet (first sheet by default) ───────────────────────
df_test = pd.read_excel(TEST_XLSX, sheet_name=0,
                        parse_dates=["ActivationDate", "MaturityDate"])

# ── loop rows & collect results ────────────────────────────────────
results = [
    inference_one_row(r, MODELS_DIR, CLIENT_NAME)
    for _, r in df_test.iterrows()
]

out_df = pd.DataFrame(results)
out_df.to_excel(OUTPUT_FILE, index=False)
print(f"✓ Saved → {OUTPUT_FILE}")
