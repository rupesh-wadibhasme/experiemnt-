# bank_features_training_xls.py
# Feature engineering + training/inference matrices for unsupervised anomaly detection (autoencoder)
# - Reads ONLY required columns from Excel
# - Cleans mixed-type categoricals (numbers/strings/blanks)
# - one_hot flag (True: OneHotEncoder, False: Ordinal/label encoding)
# - Baselines persisted as CSV (no parquet dependency)
# - Per-account asof merge, no leakage (rolling stats shifted)

from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# ========= Column map =========
DATE_COL_VALUE = "ValueDateKey"                 # YYYYMMDD int/str
DATE_COL_POST  = "PostingDateKey"               # YYYYMMDD int/str
AMOUNT_COL     = "AmountInBankAccountCurrency"  # numeric
ACCOUNT_COL    = "BankAccountCode"
CODE_COL       = "BankTransactionCode"          # mixed types & blanks -> clean
BUSUNIT_COL    = "BusinessUnitCode"
FLAG_CASHBOOK  = "CashBookFlag"                 # 'Cashbook'/'NonCashbook'/etc.
FLAG_CURRDAY   = "IsCurrentDay"                 # 0/1
MATCHED_REF    = "MatchedReference"             # 0/blank/ID

REQUIRED_COLS = [
    DATE_COL_VALUE, DATE_COL_POST, AMOUNT_COL,
    ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK,
    FLAG_CURRDAY, MATCHED_REF,
]
OPTIONAL_COLS = ["BankTransactionId"]

# IMPORTANT: AMOUNT_COL is numeric; DO NOT include it in categoricals.
CATEGORICAL_COLS = [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]

# ========= Artifacts =========
ART_DIR       = "artifacts_features"
ENCODER_PATH  = os.path.join(ART_DIR, "encoder.pkl")
ENCODER_META  = os.path.join(ART_DIR, "encoder_meta.json")  # store one_hot flag & types
SCALER_PATH   = os.path.join(ART_DIR, "standard_scaler.pkl")
BASELINE_PATH = os.path.join(ART_DIR, "account_baselines.csv")
SCHEMA_PATH   = os.path.join(ART_DIR, "feature_schema.json")

# ========= Helpers =========
def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _ensure_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

def _amount_sign(x: float) -> int:
    return int((x > 0) - (x < 0))

def _clean_categorical(s: pd.Series, name: str) -> pd.Series:
    """
    Normalize categoricals to robust strings:
    - Accept numbers/strings; cast to string
    - Strip whitespace
    - Map empty/none/nan/'0' (for refs) to UNK (domain‐specific tweak: for CashBookFlag keep value)
    """
    s = s.astype(object)
    s = s.replace({None: np.nan})
    s = s.astype(str)
    s = s.str.strip()

    # Common blanks
    blanks = {"", "nan", "NaN", "None", "NULL", "null"}
    if name == MATCHED_REF:
        # For ref we keep "0" as "0" (it may have meaning), but treat blank/nan as UNK
        s = s.map(lambda x: "UNK" if x in blanks else x)
    elif name == FLAG_CASHBOOK:
        # For flags keep raw tokens but still fix blanks
        s = s.map(lambda x: "UNK" if x in blanks else x)
    else:
        # Generic categorical (incl. BankTransactionCode)
        # Numbers become strings already; treat "0" as a valid token, blanks as UNK
        s = s.map(lambda x: "UNK" if x in blanks else x)
    return s

def _read_excel_selected(path: str, sheet_name=0) -> pd.DataFrame:
    usecols = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c not in REQUIRED_COLS]
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols=usecols,
        dtype={DATE_COL_VALUE: str, DATE_COL_POST: str},
    )
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    # Clean categoricals early (especially BankTransactionCode mixed types)
    for col in [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK, MATCHED_REF]:
        if col in df.columns:
            df[col] = _clean_categorical(df[col], col)
    return df

# ========= Row-level features =========
def engineer_row_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Numeric
    out["amount"] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").fillna(0.0)
    out["amount_log"] = np.log1p(np.abs(out["amount"]))
    out["amount_sign"] = out["amount"].apply(_amount_sign).astype("int8")

    # Dates
    val = _ensure_dt(out[DATE_COL_VALUE])
    pst = _ensure_dt(out[DATE_COL_POST])
    out["posting_lag_days"] = (pst - val).dt.days.fillna(0).astype("int16")

    # Calendar
    posting = pst.fillna(val)
    out["day_of_week"]  = posting.dt.dayofweek.astype("int8")
    out["is_weekend"]   = out["day_of_week"].isin([5, 6]).astype("int8")
    out["day_of_month"] = posting.dt.day.astype("int8")
    out["month"]        = posting.dt.month.astype("int8")
    out["quarter"]      = posting.dt.quarter.astype("int8")

    # Flags
    # Treat blanks as not matched; "0" may be a real code and was preserved by cleaner above.
    out["has_matched_ref"] = (~out[MATCHED_REF].isin(["UNK"])).astype("int8")
    out["cashbook_flag"]    = out[FLAG_CASHBOOK].str.lower().eq("cashbook").astype("int8")
    out["current_day_flag"] = pd.to_numeric(out.get(FLAG_CURRDAY, 0), errors="coerce").fillna(0).astype("int8")

    # Same-amount count within same account & posting day
    same_day = posting.dt.date
    grp = out.groupby([ACCOUNT_COL, same_day, "amount"], dropna=False).size()
    out["same_amount_count_per_day"] = grp.loc[list(zip(out[ACCOUNT_COL], same_day, out["amount"]))].values.astype("int16")

    out["ts"] = posting
    return out

# ========= Baselines (per-account, no leakage) =========
def compute_account_baselines(history_df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(history_df)
    df = df.sort_values([ACCOUNT_COL, "ts"])

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("ts")
        txn_count_7d  = g["amount"].rolling("7D").count().shift(1)
        txn_count_30d = g["amount"].rolling("30D").count().shift(1)
        mean_30d = g["amount"].rolling("30D").mean().shift(1)
        std_30d  = g["amount"].rolling("30D").std(ddof=1).shift(1)
        avg_lag_30d = g["posting_lag_days"].rolling("30D").mean().shift(1)
        return (pd.DataFrame({
            "txn_count_7d": txn_count_7d,
            "txn_count_30d": txn_count_30d,
            "mean_amount_30d": mean_30d,
            "std_amount_30d": std_30d,
            "avg_posting_lag_30d": avg_lag_30d,
        }).reset_index())

    base = df.groupby(ACCOUNT_COL, group_keys=True).apply(_roll).reset_index(level=0).rename(columns={ACCOUNT_COL: ACCOUNT_COL})
    return base

def merge_with_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(df)
    df["ts"] = pd.to_datetime(df["ts"])
    baselines["ts"] = pd.to_datetime(baselines["ts"])

    # Per-account asof
    parts = []
    for acct, g in df.groupby(ACCOUNT_COL):
        left = g.sort_values("ts")
        right = baselines[baselines[ACCOUNT_COL] == acct].sort_values("ts")
        if right.empty:
            merged = left.copy()
            merged["txn_count_7d"] = 0
            merged["txn_count_30d"] = 0
            merged["mean_amount_30d"] = 0.0
            merged["std_amount_30d"] = 0.0
            merged["avg_posting_lag_30d"] = 0.0
        else:
            merged = pd.merge_asof(left, right, on="ts", direction="backward")
        parts.append(merged)

    feats = pd.concat(parts, ignore_index=True)
    # Normalize suffixes if any
    if f"{ACCOUNT_COL}_x" in feats.columns:
        feats.rename(columns={f"{ACCOUNT_COL}_x": ACCOUNT_COL}, inplace=True)
    if f"{ACCOUNT_COL}_y" in feats.columns:
        feats.drop(columns=[f"{ACCOUNT_COL}_y"], inplace=True)

    # Deviation
    feats["zscore_amount_30d"] = (feats["amount"] - feats["mean_amount_30d"]) / feats["std_amount_30d"].replace(0, np.nan)
    feats["zscore_amount_30d"] = feats["zscore_amount_30d"].fillna(0.0).astype("float32")

    # Fill cold start
    feats[["txn_count_7d","txn_count_30d"]] = feats[["txn_count_7d","txn_count_30d"]].fillna(0)
    feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]] = feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]].fillna(0.0)
    return feats

# ========= Encoder abstraction =========
@dataclass
class FitArtifacts:
    encoder: Any
    scaler: StandardScaler
    numeric_cols: List[str]
    categorical_cols: List[str]
    all_feature_cols: List[str]
    one_hot: bool

NUMERIC_FEATURES = [
    "amount","amount_log","amount_sign","posting_lag_days","same_amount_count_per_day",
    "txn_count_7d","txn_count_30d","mean_amount_30d","std_amount_30d","avg_posting_lag_30d",
    "zscore_amount_30d","day_of_week","is_weekend","day_of_month","month","quarter",
    "has_matched_ref","cashbook_flag","current_day_flag",
]

def _fit_encoder(df_cat: pd.DataFrame, one_hot: bool):
    if one_hot:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc.fit(df_cat)
        meta = {"type": "onehot", "one_hot": True}
        names = enc.get_feature_names_out(df_cat.columns).tolist()
        return enc, meta, names
    else:
        # Label/ordinal encoding per column; unknowns -> -1
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df_cat)
        meta = {
            "type": "ordinal",
            "one_hot": False,
            "categories": [list(c) for c in enc.categories_],
            "columns": list(df_cat.columns)
        }
        # Feature names are the original categorical columns (single ordinal feature per col)
        names = list(df_cat.columns)
        return enc, meta, names

def _transform_with_encoder(enc, df_cat: pd.DataFrame, one_hot: bool) -> np.ndarray:
    return enc.transform(df_cat)

# ========= Public API =========
def fit_featureizer_from_excel(history_xlsx_path: str, sheet_name=0, one_hot: bool=True) -> FitArtifacts:
    _mkdir(ART_DIR)
    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name)

    base = compute_account_baselines(hist)
    base.to_csv(BASELINE_PATH, index=False)

    feats = merge_with_baselines(hist, base)

    # Split to numeric/cat
    X_num = feats[NUMERIC_FEATURES].astype("float32").values
    df_cat = feats[CATEGORICAL_COLS].copy()

    # Fit encoder (one-hot or ordinal)
    enc, enc_meta, cat_names = _fit_encoder(df_cat, one_hot=one_hot)
    X_cat = _transform_with_encoder(enc, df_cat, one_hot=one_hot).astype("float32")

    # Build design matrix and scaler
    X = np.hstack([X_num, X_cat]).astype("float32")
    scaler = StandardScaler().fit(X)

    # Persist
    with open(ENCODER_PATH, "wb") as f: pickle.dump(enc, f)
    with open(ENCODER_META, "w") as f: json.dump(enc_meta, f, indent=2)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    schema = {
        "numeric_cols": NUMERIC_FEATURES,
        "categorical_cols": CATEGORICAL_COLS,
        "encoded_feature_names": cat_names
    }
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)

    return FitArtifacts(
        encoder=enc,
        scaler=scaler,
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=CATEGORICAL_COLS,
        all_feature_cols=NUMERIC_FEATURES + cat_names,
        one_hot=one_hot
    )

def transform_excel_batch(batch_xlsx_path: str, sheet_name=0, one_hot: bool=True) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    # Load artifacts
    with open(ENCODER_PATH, "rb") as f: enc = pickle.load(f)
    with open(ENCODER_META, "r") as f: enc_meta = json.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler: StandardScaler = pickle.load(f)

    # Verify encoding mode matches
    if bool(enc_meta.get("one_hot", True)) != bool(one_hot):
        raise ValueError(f"Requested one_hot={one_hot} but artifacts were fit with one_hot={enc_meta.get('one_hot')}.")

    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    batch = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name)
    feats = merge_with_baselines(batch, base)

    X_num = feats[schema["numeric_cols"]].astype("float32").values
    df_cat = feats[CATEGORICAL_COLS].copy()
    X_cat = _transform_with_encoder(enc, df_cat, one_hot=one_hot).astype("float32")
    X = np.hstack([X_num, X_cat]).astype("float32")

    X_scaled = scaler.transform(X)
    feature_names = schema["numeric_cols"] + schema["encoded_feature_names"]
    return X_scaled, feats, feature_names

def update_baselines_with_excel(batch_xlsx_path: str, sheet_name=0):
    base_old = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    new_df = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name)
    base_new = compute_account_baselines(new_df)
    base_all = (pd.concat([base_old, base_new], ignore_index=True)
                  .sort_values([ACCOUNT_COL, "ts"])
                  .drop_duplicates([ACCOUNT_COL, "ts"], keep="last"))
    base_all.to_csv(BASELINE_PATH, index=False)

def build_training_matrix_from_excel(history_xlsx_path: str, sheet_name=0, one_hot: bool=True):
    """
    Build full training design matrix from 3-year history using the same encoder/scaler as inference.
    Returns: (X_scaled, feats_df, feature_names)
    Side effect: creates/overwrites artifacts in ART_DIR.
    """
    fit_featureizer_from_excel(history_xlsx_path, sheet_name=sheet_name, one_hot=one_hot)

    with open(ENCODER_PATH, "rb") as f: enc = pickle.load(f)
    with open(ENCODER_META, "r") as f: enc_meta = json.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler: StandardScaler = pickle.load(f)

    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name)
    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    feats = merge_with_baselines(hist, base)

    X_num = feats[schema["numeric_cols"]].astype("float32").values
    df_cat = feats[CATEGORICAL_COLS].copy()
    X_cat = _transform_with_encoder(enc, df_cat, one_hot=bool(enc_meta["one_hot"])).astype("float32")
    X = np.hstack([X_num, X_cat]).astype("float32")
    X_scaled = scaler.transform(X)

    feature_names = schema["numeric_cols"] + schema["encoded_feature_names"]
    return X_scaled, feats, feature_names
