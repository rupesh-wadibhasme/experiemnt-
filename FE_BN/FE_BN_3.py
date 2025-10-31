# bank_features_training_xls.py
# Feature engineering + training/inference matrices for unsupervised anomaly detection (autoencoder)
# Changes as per latest requirements:
# 1. Drop rows with duplicate BankTransactionId
# 2. has_matched_ref = 0 when MatchedReference is NULL/blank/UNK
# 3. Do NOT create day_of_week / day_of_month
# 4. Drop current_day_flag
# 5. Treat ONLY these as numeric:
#    amount,
#    posting_lag_days,
#    same_amount_count_per_day,
#    txn_count_7d,
#    txn_count_30d,
#    mean_amount_30d,
#    std_amount_30d,
#    avg_posting_lag_30d,
#    zscore_amount_30d
#    ...everything else is categorical

from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# ========= Column map =========
DATE_COL_VALUE = "ValueDateKey"                 # YYYYMMDD int/str
DATE_COL_POST  = "PostingDateKey"               # YYYYMMDD int/str
AMOUNT_COL     = "AmountInBankAccountCurrency"  # numeric
ACCOUNT_COL    = "BankAccountCode"
CODE_COL       = "BankTransactionCode"
BUSUNIT_COL    = "BusinessUnitCode"
FLAG_CASHBOOK  = "CashBookFlag"
FLAG_CURRDAY   = "IsCurrentDay"                 # will ignore
MATCHED_REF    = "MatchedReference"

REQUIRED_COLS = [
    DATE_COL_VALUE, DATE_COL_POST, AMOUNT_COL,
    ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK,
    FLAG_CURRDAY, MATCHED_REF,
]
OPTIONAL_COLS = ["BankTransactionId"]

# base categoricals (from raw data)
BASE_CATEGORICAL_COLS = [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]

# ========= Artifacts =========
ART_DIR       = "artifacts_features"
ENCODER_PATH  = os.path.join(ART_DIR, "encoder.pkl")
ENCODER_META  = os.path.join(ART_DIR, "encoder_meta.json")
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
    Normalize categoricals to robust strings.
    """
    s = s.astype(object)
    s = s.replace({None: np.nan})
    s = s.astype(str)
    s = s.str.strip()
    blanks = {"", "nan", "NaN", "None", "NULL", "null"}

    if name == MATCHED_REF:
        # ref: keep "0", blanks -> UNK
        s = s.map(lambda x: "UNK" if x in blanks else x)
    elif name == FLAG_CASHBOOK:
        s = s.map(lambda x: "UNK" if x in blanks else x)
    else:
        s = s.map(lambda x: "UNK" if x in blanks else x)
    return s

def drop_duplicate_txn_ids(df: pd.DataFrame, col: str = "BankTransactionId", keep: str = "first") -> pd.DataFrame:
    """
    Drop duplicated bank transaction IDs; keep the first occurrence.
    """
    if col not in df.columns:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=[col], keep=keep)
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before - after} duplicate {col} rows.")
    return df

def _read_excel_selected(path: str, sheet_name=0, drop_dupes: bool = True) -> pd.DataFrame:
    usecols = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c not in REQUIRED_COLS]
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols=usecols,
        dtype={DATE_COL_VALUE: str, DATE_COL_POST: str},
    )
    # required cols
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # drop dupes if present
    if drop_dupes:
        df = drop_duplicate_txn_ids(df, col="BankTransactionId", keep="first")

    # clean categoricals
    for col in [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK, MATCHED_REF]:
        if col in df.columns:
            df[col] = _clean_categorical(df[col], col)
    return df

# ========= Row-level features =========
def engineer_row_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # numeric amount
    out["amount"] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").fillna(0.0)

    # dates
    val = _ensure_dt(out[DATE_COL_VALUE])
    pst = _ensure_dt(out[DATE_COL_POST])
    out["posting_lag_days"] = (pst - val).dt.days.fillna(0).astype("int16")

    # calendar — trimmed as per requirement
    posting = pst.fillna(val)
    out["is_weekend"] = posting.dt.dayofweek.isin([5, 6]).astype("int8")
    out["month"]      = posting.dt.month.astype("int8")
    out["quarter"]    = posting.dt.quarter.astype("int8")

    # matched ref — explicit 0 for NULL/UNK
    out["has_matched_ref"] = (
        out[MATCHED_REF].notna() &
        out[MATCHED_REF].ne("UNK")
    ).astype("int8")

    # cashbook derived flag (but we’ll treat as categorical later)
    out["cashbook_flag_derived"] = out[FLAG_CASHBOOK].str.lower().eq("cashbook").astype("int8")

    # same-amount-per-day per account
    same_day = posting.dt.date
    grp = out.groupby([ACCOUNT_COL, same_day, "amount"], dropna=False).size()
    out["same_amount_count_per_day"] = grp.loc[
        list(zip(out[ACCOUNT_COL], same_day, out["amount"]))
    ].values.astype("int16")

    out["ts"] = posting
    return out

# ========= Baselines =========
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

    base = (
        df.groupby(ACCOUNT_COL, group_keys=True)
          .apply(_roll)
          .reset_index(level=0)
          .rename(columns={ACCOUNT_COL: ACCOUNT_COL})
    )
    return base

def merge_with_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(df)
    df["ts"] = pd.to_datetime(df["ts"])
    baselines["ts"] = pd.to_datetime(baselines["ts"])

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

    #normalize possible _x / _y columns after merge_asof
    if f"{ACCOUNT_COL}_x" in feats.columns:
        feats.rename(columns={f"{ACCOUNT_COL}_x": ACCOUNT_COL}, inplace=True)
    if f"{ACCOUNT_COL}_y" in feats.columns:
        feats.drop(columns=[f"{ACCOUNT_COL}_y"], inplace=True)

    # z-score
    feats["zscore_amount_30d"] = (
        (feats["amount"] - feats["mean_amount_30d"]) /
        feats["std_amount_30d"].replace(0, np.nan)
    )
    feats["zscore_amount_30d"] = feats["zscore_amount_30d"].fillna(0.0).astype("float32")

    # cold start fill
    feats[["txn_count_7d","txn_count_30d"]] = feats[["txn_count_7d","txn_count_30d"]].fillna(0)
    feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]] = (
        feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]].fillna(0.0)
    )
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

# ============ FINAL FEATURE SPLIT ============

# numeric (engineered) — as per your last message
NUMERIC_FEATURES = [
    "amount",
    "posting_lag_days",
    "same_amount_count_per_day",
    "txn_count_7d",
    "txn_count_30d",
    "mean_amount_30d",
    "std_amount_30d",
    "avg_posting_lag_30d",
    "zscore_amount_30d",
]

# categorical (raw + discrete engineered)
EXTRA_CATEGORICAL_FEATURES = [
    "is_weekend",
    "month",
    "quarter",
    "has_matched_ref",
    "cashbook_flag_derived",
]

def _fit_encoder(df_cat: pd.DataFrame, one_hot: bool):
    print('fitting encoder on categoricals')
    if one_hot:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(df_cat)
        meta = {"type": "onehot", "one_hot": True}
        names = enc.get_feature_names_out(df_cat.columns).tolist()
        return enc, meta, names
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df_cat)
        meta = {
            "type": "ordinal",
            "one_hot": False,
            "categories": [list(c) for c in enc.categories_],
            "columns": list(df_cat.columns)
        }
        names = list(df_cat.columns)
        return enc, meta, names

def _transform_with_encoder(enc, df_cat: pd.DataFrame, one_hot: bool) -> np.ndarray:
    return enc.transform(df_cat)

# ========= Public API =========

def fit_featureizer_from_excel(history_xlsx_path: str, sheet_name=0, one_hot: bool=True) -> FitArtifacts:
    _mkdir(ART_DIR)
    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)

    base = compute_account_baselines(hist)
    base.to_csv(BASELINE_PATH, index=False)

    feats = merge_with_baselines(hist, base)

    # 1) numeric
    X_num = feats[NUMERIC_FEATURES].astype("float32").copy()

    # 1a) fit scaler ONLY on amount
    amount_arr = X_num[["amount"]].values  # shape (n, 1)
    scaler = StandardScaler().fit(amount_arr)
    amount_scaled = scaler.transform(amount_arr)  # shape (n, 1)

    # other numeric (unscaled)
    other_num_cols = [c for c in NUMERIC_FEATURES if c != "amount"]
    X_other_num = X_num[other_num_cols].values  # raw values

    # 2) categorical: raw + engineered
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)

    enc, enc_meta, cat_names = _fit_encoder(df_cat, one_hot=one_hot)
    X_cat = _transform_with_encoder(enc, df_cat, one_hot=one_hot).astype("float32")

    # 3) final matrix: [amount_scaled | other_numeric_raw | encoded_cats]
    X = np.hstack([amount_scaled.astype("float32"), X_other_num.astype("float32"), X_cat]).astype("float32")

    # persist artifacts
    with open(ENCODER_PATH, "wb") as f: pickle.dump(enc, f)
    with open(ENCODER_META, "w") as f: json.dump(enc_meta, f, indent=2)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    # schema: we must remember order
    schema = {
        "numeric_cols": NUMERIC_FEATURES,  # original order
        "categorical_cols": BASE_CATEGORICAL_COLS + EXTRA_CATEGORICAL_FEATURES,
        "encoded_feature_names": cat_names
    }
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)

    return FitArtifacts(
        encoder=enc,
        scaler=scaler,
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=BASE_CATEGORICAL_COLS + EXTRA_CATEGORICAL_FEATURES,
        # final feature order == 1 scaled col + (len(NUMERIC_FEATURES)-1) raw + cats
        all_feature_cols=["amount_scaled"] + other_num_cols + cat_names,
        one_hot=one_hot
    )


def transform_excel_batch(batch_xlsx_path: str, sheet_name=0, one_hot: bool=True) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    # load artifacts
    with open(ENCODER_PATH, "rb") as f: enc = pickle.load(f)
    with open(ENCODER_META, "r") as f: enc_meta = json.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler: StandardScaler = pickle.load(f)

    if bool(enc_meta.get("one_hot", True)) != bool(one_hot):
        raise ValueError(f"Requested one_hot={one_hot} but artifacts were fit with one_hot={enc_meta.get('one_hot')}.")

    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    batch = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    feats = merge_with_baselines(batch, base)

    # numeric
    X_num = feats[schema["numeric_cols"]].astype("float32").copy()
    amount_arr = X_num[["amount"]].values
    amount_scaled = scaler.transform(amount_arr)  # shape (n,1)

    other_num_cols = [c for c in schema["numeric_cols"] if c != "amount"]
    X_other_num = X_num[other_num_cols].values  # raw

    # categorical
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)
    X_cat = _transform_with_encoder(enc, df_cat, one_hot=one_hot).astype("float32")

    # final
    X_final = np.hstack([amount_scaled.astype("float32"), X_other_num.astype("float32"), X_cat]).astype("float32")

    # feature names in the same order
    feature_names = ["amount_scaled"] + other_num_cols + schema["encoded_feature_names"]

    return X_final, feats, feature_names


def update_baselines_with_excel(batch_xlsx_path: str, sheet_name=0):
    base_old = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    new_df = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base_new = compute_account_baselines(new_df)
    base_all = (
        pd.concat([base_old, base_new], ignore_index=True)
          .sort_values([ACCOUNT_COL, "ts"])
          .drop_duplicates([ACCOUNT_COL, "ts"], keep="last")
    )
    base_all.to_csv(BASELINE_PATH, index=False)

def build_training_matrix_from_excel(history_xlsx_path: str, sheet_name=0, one_hot: bool=True):
    """
    Build full training design matrix from history using the same encoder/scaler as inference.
    """
    fit_featureizer_from_excel(history_xlsx_path, sheet_name=sheet_name, one_hot=one_hot)

    with open(ENCODER_PATH, "rb") as f: enc = pickle.load(f)
    with open(ENCODER_META, "r") as f: enc_meta = json.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler: StandardScaler = pickle.load(f)

    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    feats = merge_with_baselines(hist, base)

    X_num = feats[schema["numeric_cols"]].astype("float32").values
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)

    X_cat = _transform_with_encoder(enc, df_cat, one_hot=bool(enc_meta["one_hot"])).astype("float32")
    X_num_std = scaler.transform(X_num)
    X_final = np.hstack([X_num_std, X_cat]).astype("float32")

    feature_names = schema["numeric_cols"] + schema["encoded_feature_names"]
    return X_final, feats, feature_names
