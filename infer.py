#!/usr/bin/env python
"""
Simple inference script for the LENOVO FX-deal anomaly detector.

Run:
    python infer_anomaly.py --model ad_model_inferance_LENOVO.keras \
                            --scalers LENOVO_all_scales.pkl \
                            --test_path new_data.csv \
                            --out_path predictions.csv
"""
import argparse, json, os, pickle, warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ───────────────────────────────────────
# 1.  Utilities reused from training code
# ───────────────────────────────────────
# … data_sanity_check, feature_engg, prepare_blocks,
#    compute_errors,  get_condition_filters, get_filtered_data,
#    prompt, get_llm_output (or a stub) …
# (I’ll paste them unchanged except for log-silencing.)

# ───────────────────────────────────────
# 2.  core inference routine
# ───────────────────────────────────────
def run_inference(model_f, scalers_f, csv_f, out_f, client="LENOVO"):
    # 2-A. LOAD ARTIFACTS
    model      = tf.keras.models.load_model(model_f, compile=False)
    scaler_obj = pickle.load(open(scalers_f, "rb"))[client]
    ohe        = scaler_obj["ohe"]           # OneHotEncoder
    mms        = scaler_obj["mms"]           # MinMaxScaler (all numerics)
    fv_scalers = scaler_obj["grouped_scalers"]
    tdays_scalers = scaler_obj["tdays_scalers"]

    # 2-B. READ & PREP TEST CSV
    raw = pd.read_csv(csv_f)
    raw = data_sanity_check(raw)
    raw, num_cols, cat_cols = feature_engg(raw,
                                           numeric_columns.copy(),
                                           categorical_columns.copy())

    # one-hot on cats → *same* column order as training
    cat_ohe = pd.DataFrame(ohe.transform(raw[cat_cols]),
                           columns=ohe.get_feature_names_out(cat_cols))
    num_scaled = pd.DataFrame(mms.transform(raw[num_cols]),
                              columns=num_cols)

    feats = pd.concat([cat_ohe, num_scaled], axis=1).fillna(0)

    # 2-C. BUILD INPUT BLOCKS FOR AUTOENCODER
    cat_blocks, num_block, cardinals, _ = prepare_blocks(
        [feats.iloc[:, :cat_ohe.shape[1]],  # cats
         feats.iloc[:, cat_ohe.shape[1]:]], # nums
        numeric_block_idx=1)                # numeric block is second
    inputs = cat_blocks + [num_block]

    # ---------------------------------------------------------------
    # A.  Re-create the `features` DataFrame
    # ---------------------------------------------------------------
    features = pd.concat([cat_ohe, num_scaled], axis=1).reset_index(drop=True)
    
    # ---------------------------------------------------------------
    # B.  Slice `features` back into the 6 categorical blocks + 1 numeric block
    #     ⚠️ We rely on ohe.categories_ to know the exact width of each cat block.
    # ---------------------------------------------------------------
    cat_widths = [len(c) for c in ohe.categories_]     # e.g. [3,4,3,30,18,18]
    blocks = []
    start = 0
    for w in cat_widths:
        blocks.append(features.iloc[:, start:start + w])   # one block per cat
        start += w
    blocks.append(features.iloc[:, start:])                # numeric block
    numeric_block_idx = len(blocks) - 1                    # == 6
    
    # ---------------------------------------------------------------
    # C.  Convert blocks → model inputs
    # ---------------------------------------------------------------
    cat_arrays, num_array, _, _ = prepare_blocks(blocks, numeric_block_idx)
    inputs = cat_arrays + [num_array]   # len(inputs) == 7  ✅

    
    # 2-D. PREDICT & FLAG
    recon = model.predict(inputs, batch_size=512, verbose=0)
    _, row_score = compute_errors(cat_blocks, num_block, recon)

    ## simple flag rule from notebook
    diff = feats - pd.DataFrame(recon[-1], columns=num_cols)     # numeric diff
    anom_cnt, ok_cnt, dev_feats = get_condition_filters(diff)
    anomaly = ["Yes" if i < anom_cnt else "No" for i in range(len(raw))]

    # 2-E. LLM reason (stubbed here; replace with real call)
    reasons = []
    for idx in range(len(raw)):
        if anomaly[idx] == "Yes":
            # build LLM input & call your model
            filtered = get_filtered_data(feats.iloc[[idx]],
                                         pd.DataFrame(recon[-1], columns=num_cols).iloc[[idx]],
                                         pd.DataFrame({"stub": [0]}))   # ← put z-scores
            # llm_json = get_llm_output(str(filtered))["answer"]
            llm_json = json.dumps({"Reason for Anomaly": "⚠️ placeholder"})  # dummy
        else:
            llm_json = json.dumps({"Reason for Anomaly": "NA"})
        reasons.append(json.loads(llm_json)["Reason for Anomaly"])

    # 2-F. SAVE
    out_df = raw.copy()
    out_df["Anomaly"]           = anomaly
    out_df["reason for anomaly"] = reasons
    out_df.to_csv(out_f, index=False)
    print(f"✓ Saved predictions to {out_f}")

# ───────────────────────────────────────
# 3.  CLI
# ───────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True)
    ap.add_argument("--scalers",    required=True)
    ap.add_argument("--test_path",  required=True)
    ap.add_argument("--out_path",   default="predictions.csv")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    run_inference(args.model,
                  args.scalers,
                  args.test_path,
                  args.out_path)
