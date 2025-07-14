#!/usr/bin/env python
"""
RCC Dataset – Inference‑time pipeline
====================================
Loads the scaler/encoder artefacts saved by **data_prep.py** and transforms
**new (possibly single‑row) datasets** into the exact feature format expected
by the trained model.

The script purposefully mirrors the training‑time logic – only *drop‑duplicates*
is skipped (as per requirement).

Usage examples
--------------
    import inference_prep as ip
    artefacts = ip.load_artifacts()
    new_df = pd.read_excel("new_deals.xlsx")
    features = ip.transform_inference_df(new_df, artefacts)

    # ready for model.predict(features.values) or ip.prepare_input_data(...)
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
# 3  Static meta (copied verbatim from training script)
# ----------------------------------------------------------------------
# Columns dropped after business sign‑off
DROP_COLS = [
    'CompletionDate', 'LinkRef1', 'LinkRef2', 'LinkRef3', 'BuyOurBank',
    'BuyOurBankAccountNumber', 'SellOurBank', 'SellOurBankAccountNumber',
    'BuyTheirBank', 'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber', 'BuyBalanceMovement', 'SellBalanceMovement',
    'DealerID', 'PortCode', 'CheckedStatus', 'ComparedStatus',
    'ConfirmedStatus', 'BuyBalance', 'SellBalance', 'ForwardRate', 'DiaryDate',
    'PreparedStatus', 'BuySellTypeID', 'AuthorisedStatus', 'TranCode',
]

CAT_COLS = [
    'TranType', 'BUnit', 'Cpty', 'PrimaryCurr', 'BuyCurr', 'SellCurr',
]

NUM_COLS_ALL = [
    'BuyAmount', 'SellAmount', 'SpotRate', 'ForwardPoints',
    'Is_weekend_date', 'TDays', 'FaceValue',
]

GROUP_FACEVALUE = ['BUnit', 'Cpty', 'PrimaryCurr']
GROUP_TRANTYPE   = ['TranType']

# ----------------------------------------------------------------------
# 4  Artefact loading
# ----------------------------------------------------------------------

def load_artifacts(pkl_path: str | None = None):
    """Load artefact dict from *data_prep.py* output."""
    pkl_path = pkl_path or ARTIFACT_PKL
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    # The client key is the first (and only) key
    payload = next(iter(data.values()))
    return payload  # dict with keys mms, ohe, grouped_scalers, cpty_group, tdays_scalers

# ----------------------------------------------------------------------
# 5  Core transformation logic
# ----------------------------------------------------------------------

def _derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Weekend flag
    df['Is_weekend_date'] = df.ActivationDate.apply(lambda x: 0 if x.date().isoweekday() < 6 else 1)
    # Tenor days
    df['TDays'] = (df.MaturityDate - df.ActivationDate).dt.days
    # Ensure BUnit categorical (string)
    df['BUnit'] = df['BUnit'].astype(str)
    # FaceValue
    df['FaceValue'] = np.where(
        df.PrimaryCurr == df.BuyCurr, np.abs(df.BuyAmount),
        np.where(df.PrimaryCurr == df.SellCurr, np.abs(df.SellAmount), np.nan)
    )
    return df


def _apply_group_scaler(row: pd.Series, *, column: str, group_cols: List[str],
                         scalers: Dict) -> float:
    """Clip & scale a single value using the pre‑fitted scaler for its group."""
    # Build the group‑key exactly as saved (tuple if >1 col, else value)
    key = tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]
    if key not in scalers:
        raise ValueError(f"No scaler found for group {group_cols}={key} – retrain or expand artefacts.")
    scaler = scalers[key]['scaler']
    return float(scaler.transform([[row[column]]])[0][0])


def transform_inference_df(df_orig: pd.DataFrame, artefacts: Dict):
    """Complete feature preparation for inference."""
    # 0) Drop unused cols (if present) – duplicates NOT removed
    df = df_orig.drop(columns=[c for c in DROP_COLS if c in df_orig.columns], errors='ignore')

    # 1) Derived columns (weekend, TDays, FaceValue)
    df = _derived_columns(df)

    # 2) Clip + scale TDays (per TranType group)
    df['TDays'] = df.apply(_apply_group_scaler, axis=1,
                           column='TDays', group_cols=GROUP_TRANTYPE,
                           scalers=artefacts['tdays_scalers'])

    # 3) Clip + scale FaceValue (per BUnit‑Cpty‑PrimaryCurr group)
    df['FaceValue'] = df.apply(_apply_group_scaler, axis=1,
                               column='FaceValue', group_cols=GROUP_FACEVALUE,
                               scalers=artefacts['grouped_scalers'])

    # 4) One‑hot encode cat columns via stored encoder
    ohe: OneHotEncoder = artefacts['ohe']
    cat_df = pd.DataFrame(
        ohe.transform(df[CAT_COLS]),
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df.index,
    )

    # 5) Global MinMax scale remaining numeric columns (except the two already scaled)
    num_cols_to_scale = [c for c in NUM_COLS_ALL if c not in ('TDays', 'FaceValue')]
    mms: MinMaxScaler = artefacts['mms']
    num_df = pd.DataFrame(
        mms.transform(df[num_cols_to_scale]),
        columns=num_cols_to_scale,
        index=df.index,
    )

    # Append the per‑group scaled FaceValue & TDays
    num_df['FaceValue'] = df['FaceValue'].values
    num_df['TDays'] = df['TDays'].values

    # 6) Concat → final feature frame
    features = pd.concat([cat_df, num_df], axis=1)
    return features

# ----------------------------------------------------------------------
# 6  Optional helper to convert features → model input list
# ----------------------------------------------------------------------

def prepare_input_data(features_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[int]]:
    """Produce list‑of‑inputs + shape list (same as training helper)."""
    input_data: List[pd.DataFrame] = []
    input_shapes: List[int] = []

    for category in CAT_COLS:
        cols = [c for c in features_df.columns if c.startswith(category)]
        input_shapes.append(len(cols))
        input_data.append(features_df[cols])

    numeric_cols = [c for c in NUM_COLS_ALL]  # already present in features
    input_shapes.append(len(numeric_cols))
    input_data.append(features_df[numeric_cols])
    return input_data, input_shapes

# ----------------------------------------------------------------------
# 7  CLI entry‑point for quick testing
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="RCC inference feature builder")
    parser.add_argument("excel", help="New Excel file with raw trades")
    parser.add_argument("--artefact", default=ARTIFACT_PKL, help="Path to artefact .pkl file")
    args = parser.parse_args()

    # 1) Load artefacts + new data
    arte = load_artifacts(args.artefact)
    df_new = pd.read_excel(args.excel)
    print(f"Loaded {len(df_new):,} new rows from {args.excel}")

    # 2) Transform
    feats = transform_inference_df(df_new, arte)
    print("Feature frame shape:", feats.shape)

    # 3) (Optional) save to CSV for inspection
    csv_out = args.excel.rsplit('.', 1)[0] + "_features.csv"
    feats.to_csv(csv_out, index=False)
    print("Saved features →", csv_out)
