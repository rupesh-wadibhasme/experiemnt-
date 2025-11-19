# bank_features_simple.py
# Simple, readable feature engineering for bank-statement rows.
# - No rolling windows, no 7D/30D aggregates, no z-scores
# - Creates a stable timestamp, simple calendar features, and a combo_id
# - Outputs a tidy dataframe ready for modeling
#
# Artifacts:
#   artifacts_simple/
#     combo_mapping.json    (string combo -> integer id)
#     schema.json           (lists the numeric feature columns, etc.)

import os, json
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# ======= Config / constants =======
ART_DIR = "artifacts_simple"
COMBO_MAP_JSON = os.path.join(ART_DIR, "combo_mapping.json")
SCHEMA_JSON    = os.path.join(ART_DIR, "schema.json")

# Incoming columns we expect (you shared the Excel headers earlier)
DATE_COL_VALUE = "ValueDateKey"
DATE_COL_POST  = "PostingDateKey"
AMOUNT_COL     = "AmountInBankAccountCurrency"
ACCOUNT_COL    = "BankAccountCode"
CODE_COL       = "BankTransactionCode"
BUSUNIT_COL    = "BusinessUnitCode"
FLAG_CASHBOOK  = "CashBookFlag"

# Optional ID for duplicate dropping
TXN_ID_COL     = "BankTransactionId"

REQUIRED_COLS = [
    DATE_COL_VALUE, DATE_COL_POST, AMOUNT_COL,
    ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK
]

# ======= helpers =======
def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _ensure_dt(series: pd.Series) -> pd.Series:
    """
    Convert YYYYMMDD-like integers/strings to pandas datetime.
    Very permissive: uses exact format if possible, else fallback to to_datetime.
    """
    s = series.astype(str)
    try:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def _clean_cat(s: pd.Series) -> pd.Series:
    s = s.astype(object).replace({None: np.nan})
    s = s.astype(str).str.strip()
    blanks = {"", "nan", "NaN", "None", "NULL", "null"}
    return s.map(lambda x: "UNK" if x in blanks else x)

def _drop_dupes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[col], keep="first")
        after = len(df)
        if after < before:
            print(f"[FE] Dropped {before - after} duplicate {col} rows.")
    return df

def _same_amount_count_per_day(df: pd.DataFrame, ts_col: str, amt_col: str) -> pd.Series:
    """
    Very simple: for each (Account, BU, Code, date, amount) count occurrences.
    """
    date_only = df[ts_col].dt.date
    key = (df[ACCOUNT_COL].astype(str), df[BUSUNIT_COL].astype(str),
           df[CODE_COL].astype(str), date_only, df[amt_col])
    grp = pd.Series(1, index=key).groupby(level=[0,1,2,3,4]).sum()
    vals = grp.loc[list(zip(*key))].values
    return pd.Series(vals, index=df.index)

# ======= public API =======
def read_excel_raw(path: str, sheet_name=0, drop_dupes=True) -> pd.DataFrame:
    usecols = list(dict.fromkeys(REQUIRED_COLS + [TXN_ID_COL]))  # preserve order, include TXN_ID if present
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=[c for c in usecols if c is not None])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}\nFound: {list(df.columns)}")

    if drop_dupes:
        df = _drop_dupes(df, TXN_ID_COL)

    # clean important categoricals
    for c in [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]:
        if c in df.columns:
            df[c] = _clean_cat(df[c])

    return df

def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a very simple, readable feature set.
    - ts (timestamp): prefer PostingDateKey, fallback to ValueDateKey
    - amount (float)
    - calendar: dow, is_weekend, month, quarter
    - posting_lag_days (Posting - Value)
    - cashbook_flag_derived
    - same_amount_count_per_day
    - combo_id (string and integer)
    """
    out = df.copy()

    # timestamp
    val = _ensure_dt(out[DATE_COL_VALUE])
    pst = _ensure_dt(out[DATE_COL_POST])
    ts = pst.fillna(val)
    out["ts"] = ts

    # basic numeric
    out["amount"] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").fillna(0.0).astype(float)
    lag = (pst - val).dt.days.fillna(0).astype("int16")
    out["posting_lag_days"] = lag

    # calendar
    out["dow"]        = out["ts"].dt.dayofweek.astype("int8")
    out["is_weekend"] = out["dow"].isin([5, 6]).astype("int8")
    out["month"]      = out["ts"].dt.month.astype("int8")
    out["quarter"]    = out["ts"].dt.quarter.astype("int8")

    # flags
    out["cashbook_flag_derived"] = out[FLAG_CASHBOOK].str.lower().eq("cashbook").astype("int8")

    # same amount count per day per combo (very simple)
    out["same_amount_count_per_day"] = _same_amount_count_per_day(out, ts_col="ts", amt_col="amount").astype("int16")

    # combo id (string)
    out["combo_str"] = (
        out[ACCOUNT_COL].astype(str) + "|" +
        out[BUSUNIT_COL].astype(str) + "|" +
        out[CODE_COL].astype(str)
    )

    return out

def build_dataset_from_excel(xlsx_path: str, sheet_name=0) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Read Excel -> engineer basic features -> assign integer combo_id.
    Persist combo mapping and schema for downstream.
    Returns:
      df_feats (DataFrame with ts, amount, engineered columns, combo_id),
      tabular_feature_cols (list of numeric feature columns for the model),
      combo_map (dict: combo_str -> int id)
    """
    _mkdir(ART_DIR)
    raw = read_excel_raw(xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    feats = engineer_basic_features(raw)

    # integer ids for combos
    combos = feats["combo_str"].astype(str).unique().tolist()
    combo_map = {s: i for i, s in enumerate(sorted(combos))}
    feats["combo_id"] = feats["combo_str"].map(combo_map).astype("int32")

    # simple numeric feature set for the model (EXCLUDE 'amount' to avoid leakage)
    tabular_feature_cols = [
        "posting_lag_days",
        "dow", "is_weekend", "month", "quarter",
        "cashbook_flag_derived",
        "same_amount_count_per_day",
        # you can add more simple numerics here later if needed
    ]

    # persist artifacts
    with open(COMBO_MAP_JSON, "w") as f:
        json.dump(combo_map, f, indent=2)

    schema = {
        "tabular_feature_cols": tabular_feature_cols,
        "combo_key_cols": [ACCOUNT_COL, BUSUNIT_COL, CODE_COL],
        "amount_col": AMOUNT_COL,
        "engineered_amount_col": "amount",
        "timestamp_col": "ts",
        "combo_str_col": "combo_str",
        "combo_id_col": "combo_id"
    }
    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2)

    # return
    return feats, tabular_feature_cols, combo_map

if __name__ == "__main__":
    # tiny manual test (optional)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True)
    p.add_argument("--sheet", type=int, default=0)
    args = p.parse_args()
    df, cols, cmap = build_dataset_from_excel(args.excel, sheet_name=args.sheet)
    print(df.head())
    print("features:", cols)
    print("combos:", len(cmap))
