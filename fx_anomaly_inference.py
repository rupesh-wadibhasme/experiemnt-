"""
fx_anomaly_inference.py
–––––––––––––––––––––––
A thin wrapper around the original demo code.  Nothing in the business
rules, scaling, auto-encoder, or LLM prompt has been altered – they are
merely placed inside functions so they can be reused easily.

Public API
----------
inference_one_row(pd.Series, models_dir:str, client_name:str="RCC")
    → dict    # original columns + Anomaly + Reason for Anomaly
"""

from __future__ import annotations
import os, json, pickle, warnings
import numpy  as np
import pandas as pd
import keras
from   sklearn.preprocessing import OneHotEncoder, MinMaxScaler            # noqa
# ─────────────────────────────────────────────────────────────────────────
# 0.  suppress TF chatter
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
#               >>>>>  ORIGINAL NOTEBOOK CODE  <<<<<
# Paste *verbatim* all helpers from your demo: get_column_types,
# one_hot, scale, face_value, get_uniques, check_missing_group,
# check_currency, get_zscore, autoencode, and – most importantly –
# the massive `inference()` function that returns either a string
# (rule hit) or a tuple (...).
#
# Also paste the LLM helpers: prompt, context, get_llm_input,
# get_llm_output, get_condition_filters, get_filtered_data.
# ─────────────────────────────────────────────────────────────────────────

# (⚠️  For brevity, they are omitted here – copy-paste exactly as in
#      your long message.)


# ═══════════════════════════════════════════════════════════════════════
# 1. load_models() – unchanged helper
# ═══════════════════════════════════════════════════════════════════════
def load_models(models_path: str, client_name: str = "RCC"):
    model = keras.models.load_model(
        os.path.join(models_path, "ae-2025-06-18_RCC_f1.keras")
    )
    scalers = pickle.load(
        open(os.path.join(models_path, "all_scales.pkl"), "rb")
    )[client_name]
    return model, scalers


# ═══════════════════════════════════════════════════════════════════════
# 2.  A single-row wrapper around your original inference() & LLM logic
# ═══════════════════════════════════════════════════════════════════════
def inference_one_row(row: pd.Series,
                      models_dir: str,
                      client_name: str = "RCC",
                      rmse_threshold: float = 0.50) -> dict:
    """
    Parameters
    ----------
    row          : one deal (pd.Series)
    models_dir   : folder containing .keras & .pkl
    client_name  : key for scaler bundle in pickle
    rmse_threshold : cut-off used inside get_response()

    Returns
    -------
    dict  – all original columns plus:
            • 'Anomaly'            : 'Yes' / 'No'
            • 'Reason for Anomaly' : str (may be 'NA')
    """
    # 0️⃣  call your untouched `inference()` – it **expects a DataFrame**
    scores_or_msg, *rest = inference(
        pd.DataFrame([row]), models_dir, client_name
    )

    # case A: rule-based anomaly (inference returned a string)
    if isinstance(scores_or_msg, str):
        return {**row,
                "Anomaly": "Yes",
                "Reason for Anomaly": scores_or_msg}

    # case B: we got the full tuple back
    scores, features, _, recon_df, _, zscores = (
        scores_or_msg, *rest)  # unpack with original order

    # Use your demo’s helper to decide severity & LLM
    filtered = get_filtered_data(features, recon_df, zscores)
    if len(filtered["Deviated_Features"]) == 0:
        # model thinks it’s fine
        return {**row, "Anomaly": "No", "Reason for Anomaly": "NA"}

    llm_input  = str(filtered)
    llm_answer = get_llm_output(llm_input)["answer"]
    try:
        reason = json.loads(llm_answer)["Reason for Anomaly"]
    except Exception:
        reason = "LLM explanation unavailable"

    return {**row,
            "Anomaly": "Yes",
            "Reason for Anomaly": reason}
