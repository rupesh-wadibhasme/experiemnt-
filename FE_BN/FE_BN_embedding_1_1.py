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
from typing import List, Tuple, Dict, Union
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
def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

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

def _clean_raw_df(df: pd.DataFrame, drop_dupes: bool = True) -> pd.DataFrame:
    """
    Common cleaning logic for both Excel and DataFrame inputs:
      - check required cols
      - drop dupes (if requested)
      - clean categoricals
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

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

# ======= public API: reading =======
def read_excel_raw(path: str, sheet_name=0, drop_dupes=True) -> pd.DataFrame:
    usecols = list(dict.fromkeys(REQUIRED_COLS + [TXN_ID_COL]))  # preserve order
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=[c for c in usecols if c is not None])
    return _clean_raw_df(df, drop_dupes=drop_dupes)

# ======= feature engineering =======
def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a very simple, readable feature set.
    - ts (timestamp): prefer PostingDateKey, fallback to ValueDateKey
    - amount (float)
    - calendar: dow, is_weekend, month, quarter, cyclical encodings
    - posting_lag_days (Posting - Value)
    - cashbook_flag_derived
    - log_amount (signed log1p)
    - combo_str (Account|BU|Code)
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
    out["dow"]        = out["ts"].dt.dayofweek.astype("int8")   # 0=Mon
    out["is_weekend"] = out["dow"].isin([5, 6]).astype("int8")
    out["month"]      = out["ts"].dt.month.astype("int8")
    out["quarter"]    = out["ts"].dt.quarter.astype("int8")

    # simple cyclical encodings
    m = out["month"].astype(float)
    out["month_sin"] = np.sin(2 * np.pi * (m / 12.0))
    out["month_cos"] = np.cos(2 * np.pi * (m / 12.0))
    dw = out["dow"].astype(float)
    out["dow_sin"]   = np.sin(2 * np.pi * (dw / 7.0))
    out["dow_cos"]   = np.cos(2 * np.pi * (dw / 7.0))

    # flags
    out["cashbook_flag_derived"] = out[FLAG_CASHBOOK].str.lower().eq("cashbook").astype("int8")

    # signed log amount
    out["log_amount"] = np.sign(out["amount"]) * np.log1p(np.abs(out["amount"]))

    # combo id (string)
    out["combo_str"] = (
        out[ACCOUNT_COL].astype(str) + "|" +
        out[BUSUNIT_COL].astype(str) + "|" +
        out[CODE_COL].astype(str)
    )
    # ---------- New addition -------
    # daily transaction count per (Account, BU, Code, ValueDateKey)
    group_cols = [ACCOUNT_COL, BUSUNIT_COL, CODE_COL, DATE_COL_VALUE]
    out["trans_count_day"] = (
        out.groupby(group_cols)[AMOUNT_COL].transform("size").astype("int32")
    )
    out["trans_count_log"] = np.log1p(out["trans_count_day"].astype("float32"))


    return out

# ======= core builder (common) =======
def _build_dataset_common(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Shared logic: takes a cleaned raw df, engineers features, builds combo_id,
    persists artifacts, returns (feats, tabular_feature_cols, combo_map).
    """
    _mkdir(ART_DIR)
    feats = engineer_basic_features(raw)

    # integer ids for combos
    combos = feats["combo_str"].astype(str).unique().tolist()
    combo_map = {s: i for i, s in enumerate(sorted(combos))}
    feats["combo_id"] = feats["combo_str"].map(combo_map).astype("int32")

    # simple numeric feature set for the model (EXCLUDE 'amount' to avoid target leakage)
    # keep this explicit & easy to read
    candidate_tab_cols = [
        "posting_lag_days",
        "dow", "is_weekend", "month", "quarter",
        "cashbook_flag_derived",
        "month_sin", "month_cos",
        "dow_sin", "dow_cos",
        "log_amount",
    ]
    tabular_feature_cols = [c for c in candidate_tab_cols if c in feats.columns]

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
        "combo_id_col": "combo_id",
    }
    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2)

    # sort by time to make any time-based split stable
    feats = feats.sort_values("ts").reset_index(drop=True)

    return feats, tabular_feature_cols, combo_map

# ======= public builders =======
def build_dataset_from_df(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Main entry when you already have a pandas DataFrame in memory.
    """
    raw = _clean_raw_df(df_raw, drop_dupes=True)
    return _build_dataset_common(raw)

def build_dataset_from_excel(x: Union[str, os.PathLike, pd.DataFrame], sheet_name=0) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Backward-compatible:
      - If x is a path-like, read the Excel.
      - If x is already a DataFrame, treat it as raw input.
    """
    if isinstance(x, pd.DataFrame):
        raw = _clean_raw_df(x, drop_dupes=True)
    else:
        raw = read_excel_raw(x, sheet_name=sheet_name, drop_dupes=True)
    return _build_dataset_common(raw)

# ======= tiny manual test (optional) =======
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=False, help="Path to Excel (optional)")
    p.add_argument("--sheet", type=int, default=0)
    args = p.parse_args()

    if args.excel:
        df_feats, cols, cmap = build_dataset_from_excel(args.excel, sheet_name=args.sheet)
        print(df_feats.head())
        print("tabular_feature_cols:", cols)
        print("num combos:", len(cmap))
    else:
        print("bank_features_simple.py ready. Use build_dataset_from_df(...) or build_dataset_from_excel(...).")
