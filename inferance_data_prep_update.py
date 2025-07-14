#!/usr/bin/env python
"""
RCC Dataset – Inference‑time pipeline (v1.1)
===========================================
*Patch 14 Jul 2025:*
  • **Fixed group‑key lookup** when the stored scaler dictionary uses 1‑element tuples
    (pandas `groupby([...])` behaviour). The helper now builds a tuple key and
    falls back to a scalar lookup, eliminating the "No scaler found" error.

Loads the scaler/encoder artefacts saved by **data_prep.py** and transforms
new datasets (single row or batch) into the exact feature format expected by
your model.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# 1  EDIT PATHS HERE ONLY
# ----------------------------------------------------------------------
ARTIFACT_PKL = r"C:\PATH\TO\ARTIFACTS\pickle files\all_scales_NonEmbed_<timestamp>.pkl"

# ----------------------------------------------------------------------
# 2  Imports
# ----------------------------------------------------------------------
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# ----------------------------------------------------------------------
# 3  Static meta (identical to training script)
# ----------------------------------------------------------------------
DROP_COLS = [
    'CompletionDate', 'LinkRef1', 'LinkRef2', 'LinkRef3', 'BuyOurBank',
    'BuyOurBankAccountNumber', 'SellOurBank', 'SellOurBankAccountNumber',
    'BuyTheirBank', 'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber', 'BuyBalanceMovement', 'SellBalanceMovement',
    'DealerID', 'PortCode', 'CheckedStatus', 'ComparedStatus',
    'ConfirmedStatus', 'BuyBalance', 'SellBalance', 'ForwardRate', 'DiaryDate',
    'PreparedStatus', 'BuySellTypeID', 'AuthorisedStatus', 'TranCode',
]

CAT_COLS = ['Instrument', 'BUnit', 'Cpty', 'PrimaryCurr', 'BuyCurr', 'SellCurr']

NUM_COLS_ALL = [
    'BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints',
    'Is_weekend_date', 'TDays', 'FaceValue',
]

GROUP_FACEVALUE = ['BUnit', 'Cpty', 'PrimaryCurr']
GROUP_TRANTYPE  = ['Instrument']

# ----------------------------------------------------------------------
# 4  Artefact loading
# ----------------------------------------------------------------------

def load_artifacts(pkl_path: str | None = None):
    """Load artefact dict produced by *data_prep.py*."""
    pkl_path = pkl_path or ARTIFACT_PKL
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    return next(iter(data.values()))  # {'mms':…, 'ohe':…, …}

# ----------------------------------------------------------------------
# 5  Core transformation logic
# ----------------------------------------------------------------------

def _derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Is_weekend_date'] = df.ActivationDate.apply(lambda x: 0 if x.date().isoweekday() < 6 else 1)
    df['TDays'] = (df.MaturityDate - df.ActivationDate).dt.days
    df['BUnit'] = df['BUnit'].astype(str)
    df['Instrument'] = df['Instrument'].astype(str).str.strip().str.upper()
    df['FaceValue'] = np.where(
        df.PrimaryCurr == df.BuyCurr, np.abs(df.BuyAmount),
        np.where(df.PrimaryCurr == df.SellCurr, np.abs(df.SellAmount), np.nan)
    )
    return df


def _apply_group_scaler(row: pd.Series, *, column: str, group_cols: List[str],
                         scalers: Dict) -> float:
    """Scale a value with the pre‑fitted group scaler (handles 1‑col tuple keys)."""
    key_tuple = tuple(row[col] for col in group_cols)

    # First try tuple key (('FXFORWARD',), …)
    scaler_entry = scalers.get(key_tuple)
    # Fallback: scalar key ('FXFORWARD') if the training artefact stored scalars
    if scaler_entry is None and len(key_tuple) == 1:
        scaler_entry = scalers.get(key_tuple[0])

    if scaler_entry is None:
        sample_keys = list(scalers.keys())[:10]
        raise ValueError(
            f"No scaler found for group {group_cols}={key_tuple}.\n"
            f"Available keys (first 10): {sample_keys}"
        )

    scaler = scaler_entry['scaler']
    return float(scaler.transform([[row[column]]])[0, 0])


def transform_inference_df(df_raw: pd.DataFrame, artefacts: Dict):
    """Convert raw dataframe to model‑ready feature frame."""
    df = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns], errors='ignore')
    df = _derived_columns(df)

    df['TDays'] = df.apply(_apply_group_scaler, axis=1,
                           column='TDays', group_cols=GROUP_TRANTYPE,
                           scalers=artefacts['tdays_scalers'])

    df['FaceValue'] = df.apply(_apply_group_scaler, axis=1,
                               column='FaceValue', group_cols=GROUP_FACEVALUE,
                               scalers=artefacts['grouped_scalers'])

    ohe: OneHotEncoder = artefacts['ohe']
    cat_df = pd.DataFrame(
        ohe.transform(df[CAT_COLS]),
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df.index,
    )

    mms: MinMaxScaler = artefacts['mms']
    to_scale = [c for c in NUM_COLS_ALL if c not in ('TDays', 'FaceValue')]
    num_df = pd.DataFrame(
        mms.transform(df[to_scale]),
        columns=to_scale,
        index=df.index,
    )
    num_df['FaceValue'] = df['FaceValue'].values
    num_df['TDays'] = df['TDays'].values

    return pd.concat([cat_df, num_df], axis=1)

# ----------------------------------------------------------------------
# 6  Optional helper – model input list
# ----------------------------------------------------------------------

def prepare_input_data(features_df: pd.DataFrame):
    input_data, shapes = [], []
    for cat in CAT_COLS:
        cols = [c for c in features_df.columns if c.startswith(cat)]
        shapes.append(len(cols))
        input_data.append(features_df[cols])
    shapes.append(len(NUM_COLS_ALL))
    input_data.append(features_df[NUM_COLS_ALL])
    return input_data, shapes

# ----------------------------------------------------------------------
# 7  CLI for ad‑hoc use
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="RCC inference feature builder")
    parser.add_argument("excel", help="Excel file with raw trades")
    parser.add_argument("--artefact", default=ARTIFACT_PKL)
    args = parser.parse_args()

    arte = load_artifacts(args.artefact)
    df_new = pd.read_excel(args.excel)
    print(f"Loaded {len(df_new):,} rows from {args.excel}")

    feats = transform_inference_df(df_new, arte)
    print("Feature frame shape:", feats.shape)
    out_csv = args.excel.rsplit('.', 1)[0] + "_features.csv"
    feats.to_csv(out_csv, index=False)
    print("Saved features →", out_csv)
