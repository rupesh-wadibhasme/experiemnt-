#!/usr/bin/env python
"""
RCC Dataset – Data‑preparation pipeline
======================================
Refactored from the original Jupyter notebook, 14 July 2025.
*All business logic and heuristics are preserved; the code is only reorganised for clarity.*

Update the *PATH_* constants once and run the script – no other changes required.
The script is intentionally kept simple (some inefficiencies remain) to mirror the
notebook’s behaviour exactly.
"""

# %% ---------------------------------------------------------------------
# Standard library & third‑party imports
# ----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import pickle
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401  – kept for parity with notebook
import seaborn as sns             # noqa: F401  – kept for parity with notebook
import kneed                      # noqa: F401  – kept for parity with notebook

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# %% ---------------------------------------------------------------------
# -----------------------------------------------------------------------
# 1  Configuration – edit these paths in one place only
# -----------------------------------------------------------------------
# NOTE: Use raw‑string literals if you are on Windows.
DATA_PATH   = r"C:\PATH\TO\DATA\Copy of RCC_distinct_28052025.xlsx"
MODELS_PATH = r"C:\PATH\TO\ARTIFACTS\models"
PICKLES_PATH = r"C:\PATH\TO\ARTIFACTS\pickle files"
CLIENT_NAME = "RCC"

# %% ---------------------------------------------------------------------
# 2  Static column lists (verbatim from notebook)
# -----------------------------------------------------------------------
master_cols_list = [
    'TranCode', 'TranType', 'BUnit', 'Cpty',
    'ActivationDate', 'CompletionDate', 'DiaryDate', 'MaturityDate',
    'PrimaryCurr', 'BuySellTypeID', 'AuthorisedStatus', 'CheckedStatus',
    'ComparedStatus', 'PreparedStatus', 'ConfirmedStatus', 'BuyAmount',
    'SellAmount', 'BuyCurr', 'SellCurr', 'BuyBalanceMovement',
    'SellBalanceMovement', 'BuyBalance', 'SellBalance', 'SpotRate',
    'ForwardPoints', 'ForwardRate', 'PortCode', 'DealerID', 'LinkRef1',
    'LinkRef2', 'LinkRef3', 'BuyOurBank', 'BuyOurBankAccountNumber',
    'SellOurBank', 'SellOurBankAccountNumber', 'BuyTheirBank',
    'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber',
]

# Columns dropped after business sign‑off (2025‑06‑11)
drop_cols_master = [
    'CompletionDate', 'LinkRef1', 'LinkRef2', 'LinkRef3', 'BuyOurBank',
    'BuyOurBankAccountNumber', 'SellOurBank', 'SellOurBankAccountNumber',
    'BuyTheirBank', 'BuyTheirBankAccountNumber', 'SellTheirBank',
    'SellTheirBankAccountNumber', 'BuyBalanceMovement', 'SellBalanceMovement',
    'DealerID', 'PortCode', 'CheckedStatus', 'ComparedStatus',
    'ConfirmedStatus', 'BuyBalance', 'SellBalance', 'ForwardRate', 'DiaryDate',
    'PreparedStatus', 'BuySellTypeID', 'AuthorisedStatus',  # dropped 06‑Jun‑25
    'TranCode',                                             # dropped 11‑Jun‑25
]

# -----------------------------------------------------------------------
# 3  Utility functions
# -----------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read the Excel file and filter out rows with zero Buy/Sell amounts."""
    df = pd.read_excel(path, header=0)
    df = df[(df.BuyAmount != 0.0) & (df.SellAmount != 0.0)]
    print(f"Loaded {len(df):,} rows (non‑zero Buy/Sell amounts).")
    return df.copy()


def drop_unwanted_data(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """Drop agreed‑upon columns and duplicate rows (keep last)."""
    df = df.drop(columns=cols_to_drop, errors="ignore")
    before = len(df)
    df = df.drop_duplicates(keep="last")
    print(f"Drop duplicates/cols → size {before:,} → {len(df):,} rows.")
    return df


# ......................................................................
# Derived‑feature engineering
# ......................................................................

def add_derived_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Create Is_weekend_date, TDays, FaceValue; cast BUnit categorical."""
    if 'Factor' in df.columns:
        df = df.drop(columns=['Factor'])

    # Weekend indicator (Mon=1 – Sun=7 → weekend = 1)
    df['Is_weekend_date'] = df.ActivationDate.apply(lambda x: 0 if x.date().isoweekday() < 6 else 1)

    # Tenor in days
    df['TDays'] = (df.MaturityDate - df.ActivationDate).dt.days

    # Ensure BUnit numeric treated as category
    df['BUnit'] = df['BUnit'].astype(str)

    # FaceValue based on Primary currency
    df['FaceValue'] = np.where(
        df.PrimaryCurr == df.BuyCurr, np.abs(df.BuyAmount),
        np.where(df.PrimaryCurr == df.SellCurr, np.abs(df.SellAmount), np.nan)
    )

    # Column groups
    categorical_columns = [
        'TranType', 'BUnit', 'Cpty',
        'PrimaryCurr', 'BuyCurr', 'SellCurr',
    ]
    numeric_columns = [
        'BuyAmount', 'SellAmount', 'SpotRate',
        'ForwardPoints', 'Is_weekend_date', 'TDays', 'FaceValue',
    ]
    return df, categorical_columns, numeric_columns


# ......................................................................
# Group‑wise clipping and scaling helpers (verbatim logic)
# ......................................................................
from sklearn.preprocessing import MinMaxScaler  # (imported again for clarity)

def _scale_group(group: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, MinMaxScaler, float, float]:
    scaler = MinMaxScaler()
    group[[column]] = scaler.fit_transform(group[[column]])
    return group, scaler, float(group[column].mean()), float(group[column].std())


def _clip_percentile(group: pd.DataFrame, column: str, upp: float = 0.99, low: float = 0.01):
    if len(group[column]) < 10:
        lower, upper = group[column].min(), group[column].max()
        counts = (0, 0, 0)
    else:
        upper, lower = group[column].quantile(upp), group[column].quantile(low)
        lower_counts = (group[column] < lower).sum()
        upper_counts = (group[column] > upper).sum()
        counts = ((~group[column].between(lower, upper)).sum(), lower_counts, upper_counts)
        group[column] = group[column].clip(lower, upper)
    return group, lower, upper, counts


def group_points(df: pd.DataFrame, group_cols: List[str], column: str,
                 *, percentile: bool = True,
                 need_scaled_df: bool = True):
    """Apply percentile clipping + MinMax scaling within groups."""
    grouped_scalers: Dict = {}
    scaled_groups = []
    total_counts = [0, 0, 0]

    for name, grp in df.groupby(group_cols, sort=False):
        if percentile:
            grp, *_counts = _clip_percentile(grp, column)
        else:
            raise NotImplementedError("Only percentile mode retained to match notebook")

        # tally clip counts
        total_counts = [x + y for x, y in zip(total_counts, _counts[2])]

        if need_scaled_df:
            grp, scaler, mean, sd = _scale_group(grp, column)
            grouped_scalers[name] = {'scaler': scaler, 'mean': mean, 'sd': sd}
        scaled_groups.append(grp)

    return pd.concat(scaled_groups), grouped_scalers, tuple(total_counts)


# ......................................................................
# Encoding & final feature frame
# ......................................................................

global_scaler: MinMaxScaler | None = None  # keeps behaviour identical to notebook


def one_hot(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df)
    enc = encoder.transform(df)
    return pd.DataFrame(enc, columns=encoder.get_feature_names_out(df.columns)), encoder


def minmax_scale(df: pd.DataFrame):
    global global_scaler
    if global_scaler is None:
        global_scaler = MinMaxScaler()
    global_scaler.fit(df)
    scaled = global_scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns).fillna(0), global_scaler


def build_feature_frame(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]):
    # Remove already‑scaled FaceValue/TDays for the generic scaler
    num_cols_ = [c for c in num_cols if c not in ('FaceValue', 'TDays')]

    cat_df, ohe = one_hot(df[cat_cols])
    num_df, mms = minmax_scale(df[num_cols_])

    # append scaled FaceValue & TDays directly (already 0‑1)
    num_df['FaceValue'] = df['FaceValue'].values
    num_df['TDays'] = df['TDays'].values

    # track index then drop helper column
    cat_df['Index'] = df.index
    features = pd.concat([cat_df, num_df], axis=1)
    features.drop(columns='Index', inplace=True)
    features['FaceValue'].fillna(0, inplace=True)
    return features, ohe, mms


# ......................................................................
# Misc helpers
# ......................................................................

def build_cpty_groups(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for cp in df['Cpty'].unique().tolist():
        out[cp] = {
            'buy': df[df['Cpty'] == cp]['BuyCurr'].unique().tolist(),
            'sell': df[df['Cpty'] == cp]['SellCurr'].unique().tolist(),
        }
    return out


def save_artifacts(mms: MinMaxScaler, ohe: OneHotEncoder,
                   fv_scalers: Dict, cpty_groups: Dict,
                   tdays_scalers: Dict, *, dst: str):
    os.makedirs(dst, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    fname = f"all_scales_NonEmbed_{mms}_{ts}.pkl".replace(' ', '')
    payload = {
        CLIENT_NAME: {
            'mms': mms,
            'ohe': ohe,
            'grouped_scalers': fv_scalers,
            'cpty_group': cpty_groups,
            'tdays_scalers': tdays_scalers,
        }
    }
    with open(os.path.join(dst, fname), 'wb') as fh:
        pickle.dump(payload, fh)
    print(f"Artifacts saved → {fh.name}")


# ......................................................................
# Input‑tensor helper (retained verbatim)
# ......................................................................

def prepare_input_data(features_df: pd.DataFrame,
                        categorical_columns: List[str],
                        numeric_columns: List[str]):
    """Return list‑of‑arrays for model.consume() + per‑input shapes."""
    input_data: List[pd.DataFrame] = []
    input_shapes: List[int] = []

    for category in categorical_columns:
        cols = [c for c in features_df.columns if c.startswith(category)]
        input_shapes.append(len(cols))
        input_data.append(features_df[cols])

    input_shapes.append(len(numeric_columns))
    input_data.append(features_df[numeric_columns])
    return input_data, input_shapes


# %% ---------------------------------------------------------------------
# -----------------------------------------------------------------------
# 4  Main routine
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # 0) Load raw data
    raw = load_data(DATA_PATH)

    # 1) Drop unwanted columns / duplicates
    data = drop_unwanted_data(raw, drop_cols_master)

    # 2) Feature engineering
    data, cat_cols, num_cols = add_derived_features(data)

    # 3) Group‑wise clipping+scaling (percentile method)
    group_facevalue = ['BUnit', 'Cpty', 'PrimaryCurr']
    group_trantype = ['TranType']

    grouped_df, tdays_scalers, _ = group_points(data, group_trantype, 'TDays')
    grouped_df, fv_scalers, _    = group_points(grouped_df, group_facevalue, 'FaceValue')

    # 4) Save convenience index if needed downstream
    group_clipped_idx = grouped_df.index

    # 5) Counter‑party currency groups
    cpty_groups = build_cpty_groups(data)

    # 6) One‑hot + MinMax scaling → final feature dataframe
    features, ohe, mms = build_feature_frame(grouped_df, cat_cols, num_cols)

    # 7) Persist scalers & metadata
    save_artifacts(mms, ohe, fv_scalers, cpty_groups, tdays_scalers, dst=PICKLES_PATH)

    # 8) Optionally build nn‑ready tensors (train/test split not included here)
    #     – replicate notebook logic outside if desired.
    # train_input, input_shapes = prepare_input_data(features, cat_cols, num_cols)
    print("Data‑preparation pipeline complete.")
