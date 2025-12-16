# bank_features_simple.py  (FE_BN_embeding_1_1)
# Simple, readable feature engineering for bank-statement rows.
# - No rolling windows, no 7D/30D aggregates, no z-scores
# - Creates a stable timestamp, simple calendar features, and a combo_id
# - Outputs a tidy dataframe ready for modeling
#
# Artifacts:
#   artifacts_simple/
#     combo_mapping.json
#     account_mapping.json
#     businessunit_mapping.json
#     code_mapping.json
#     schema.json

import os, json
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd

# ======= Config / constants =======
ART_DIR = "artifacts_simple"
COMBO_MAP_JSON = os.path.join(ART_DIR, "combo_mapping.json")
SCHEMA_JSON    = os.path.join(ART_DIR, "schema.json")

# NEW: per-column categorical mappings (for separate embeddings)
ACCOUNT_MAP_JSON = os.path.join(ART_DIR, "account_mapping.json")
BUSUNIT_MAP_JSON = os.path.join(ART_DIR, "businessunit_mapping.json")
CODE_MAP_JSON    = os.path.join(ART_DIR, "code_mapping.json")

# OOV token for categorical ids
OOV_TOKEN = "__OOV__"

# Incoming columns we expect
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

# Numerics safety
EPS_MEAN = 1e-6

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

def _make_id_map(values: pd.Series) -> Dict[str, int]:
    """
    Stable mapping: sorted unique strings, with OOV_TOKEN reserved as last id.
    """
    uniq = sorted(values.astype(str).unique().tolist())
    m = {v: i for i, v in enumerate(uniq)}
    if OOV_TOKEN not in m:
        m[OOV_TOKEN] = len(m)
    return m

def _apply_id_map(values: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    oov_id = mapping[OOV_TOKEN]
    return values.astype(str).map(mapping).fillna(oov_id).astype("int32").values

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

    NEW additions:
    - trans_count_day, trans_count_log (per Account+BU+Code per ValueDateKey)
    - group-context features (computed within combo-day):
        * amount_dev_loo_log
        * amount_ratio_loo_slog
        * identical_amounts_in_group
        * single_transaction_group
        * amount_sign
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

    # combo key (string)
    out["combo_str"] = (
        out[ACCOUNT_COL].astype(str) + "|" +
        out[BUSUNIT_COL].astype(str) + "|" +
        out[CODE_COL].astype(str)
    )

    # --- daily transaction count per (Account, BU, Code, ValueDateKey) ---
    group_cols = [ACCOUNT_COL, BUSUNIT_COL, CODE_COL, DATE_COL_VALUE]
    out["trans_count_day"] = out.groupby(group_cols)[AMOUNT_COL].transform("size").astype("int32")
    out["trans_count_log"] = np.log1p(out["trans_count_day"].astype("float32"))

    # --- group-context features within combo-day ---
    # Define combo-day group using the same keys as trans_count_day (stable and explicit)
    amt = out["amount"].astype("float64").values
    gsize = out["trans_count_day"].astype("int32").values

    # group sum (within combo-day)
    group_sum = out.groupby(group_cols)["amount"].transform("sum").astype("float64").values

    # leave-one-out mean:
    # mean_loo = (sum - amt) / (n-1) if n>1 else (sum/n)
    denom = np.maximum(gsize - 1, 1).astype("float64")
    mean_loo = (group_sum - amt) / denom
    # if single txn, mean_loo becomes (sum-amt)/1 = 0; better fallback to group mean
    group_mean = group_sum / np.maximum(gsize, 1).astype("float64")
    mean_loo = np.where(gsize > 1, mean_loo, group_mean)

    # deviation from leave-one-out mean (log-scaled for stability)
    dev_loo = np.abs(amt - mean_loo)
    out["amount_dev_loo_log"] = np.log1p(dev_loo).astype("float32")

    # ratio to leave-one-out mean (signed-log for stability)
    safe_mean = np.where(np.abs(mean_loo) < EPS_MEAN, np.sign(mean_loo) * EPS_MEAN + EPS_MEAN, mean_loo)
    ratio_loo = amt / safe_mean
    out["amount_ratio_loo_slog"] = (np.sign(ratio_loo) * np.log1p(np.abs(ratio_loo))).astype("float32")

    # identical amounts within group (all amounts same)
    nunique_amt = out.groupby(group_cols)["amount"].transform("nunique").astype("int32").values
    out["identical_amounts_in_group"] = (nunique_amt == 1).astype("int8")

    # single transaction group
    out["single_transaction_group"] = (gsize == 1).astype("int8")

    # amount sign (-1, 0, +1)
    out["amount_sign"] = np.sign(out["amount"]).astype("int8")

    return out

# ======= core builder (common) =======
def _build_dataset_common(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Shared logic: takes a cleaned raw df, engineers features, builds combo_id,
    persists artifacts, returns (feats, tabular_feature_cols, combo_map).
    """
    _mkdir(ART_DIR)
    feats = engineer_basic_features(raw)

    # integer ids for combos (stable sorted)
    combos = feats["combo_str"].astype(str).unique().tolist()
    combo_map = {s: i for i, s in enumerate(sorted(combos))}
    feats["combo_id"] = feats["combo_str"].map(combo_map).astype("int32")

    # NEW: separate id maps for account / BU / code (stable sorted + OOV)
    acct_map = _make_id_map(feats[ACCOUNT_COL])
    bu_map   = _make_id_map(feats[BUSUNIT_COL])
    code_map = _make_id_map(feats[CODE_COL])

    feats["account_id"] = _apply_id_map(feats[ACCOUNT_COL], acct_map)
    feats["bu_id"]      = _apply_id_map(feats[BUSUNIT_COL], bu_map)
    feats["code_id"]    = _apply_id_map(feats[CODE_COL], code_map)

    # --- numeric feature set for the model ---
    # keep this explicit & easy to read
    #
    # NOTE:
    # - We keep trans_count_day/log available in the dataframe for downstream count_norm logic.
    # - We do NOT automatically add trans_count_log to tabular features here, because your
    #   training script may prefer feeding only count_norm (train-only normalized) into the AE.
    candidate_tab_cols = [
        "posting_lag_days",
        "dow", "is_weekend", "month", "quarter",
        "cashbook_flag_derived",
        "month_sin", "month_cos",
        "dow_sin", "dow_cos",
        "log_amount",

        # NEW group-context features (safe, bounded)
        "amount_dev_loo_log",
        "amount_ratio_loo_slog",
        "identical_amounts_in_group",
        "single_transaction_group",
        "amount_sign",
    ]
    tabular_feature_cols = [c for c in candidate_tab_cols if c in feats.columns]

    # persist artifacts
    with open(COMBO_MAP_JSON, "w") as f:
        json.dump(combo_map, f, indent=2)

    with open(ACCOUNT_MAP_JSON, "w") as f:
        json.dump(acct_map, f, indent=2)

    with open(BUSUNIT_MAP_JSON, "w") as f:
        json.dump(bu_map, f, indent=2)

    with open(CODE_MAP_JSON, "w") as f:
        json.dump(code_map, f, indent=2)

    schema = {
        "tabular_feature_cols": tabular_feature_cols,
        "combo_key_cols": [ACCOUNT_COL, BUSUNIT_COL, CODE_COL],
        "amount_col": AMOUNT_COL,
        "engineered_amount_col": "amount",
        "timestamp_col": "ts",
        "combo_str_col": "combo_str",
        "combo_id_col": "combo_id",

        # NEW: categorical id cols for separate embeddings
        "account_id_col": "account_id",
        "bu_id_col": "bu_id",
        "code_id_col": "code_id",
        "oov_token": OOV_TOKEN,
        "account_map_json": os.path.basename(ACCOUNT_MAP_JSON),
        "businessunit_map_json": os.path.basename(BUSUNIT_MAP_JSON),
        "code_map_json": os.path.basename(CODE_MAP_JSON),

        # NEW: daily-count columns (available for count_norm logic)
        "count_day_cols": group_cols if False else [ACCOUNT_COL, BUSUNIT_COL, CODE_COL, DATE_COL_VALUE],
        "trans_count_day_col": "trans_count_day",
        "trans_count_log_col": "trans_count_log",

        # NEW: group-context cols (for interpretability)
        "group_context_cols": [
            "amount_dev_loo_log",
            "amount_ratio_loo_slog",
            "identical_amounts_in_group",
            "single_transaction_group",
            "amount_sign",
        ],
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
        print("has account_id/bu_id/code_id:", {"account_id","bu_id","code_id"}.issubset(df_feats.columns))
    else:
        print("bank_features_simple.py ready. Use build_dataset_from_df(...) or build_dataset_from_excel(...).")
