# bank_features_training_xls.py
# Hybrid feature engineering for unsupervised anomaly detection.
# - Numeric features (configurable subset scaled)
# - Categorical:
#     * auto-detect cardinality per column on fit
#     * <= MAX_OHE_CARD -> OneHot
#     *  > MAX_OHE_CARD -> Hashing into HASH_BINS bins (per column)
# - Persist artifacts (scaler + hybrid encoder + schema)
# - Baselines: rolling 7D/30D per account (no leakage)

from __future__ import annotations
import os, json, pickle, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ========= Columns =========
DATE_COL_VALUE = "ValueDateKey"
DATE_COL_POST  = "PostingDateKey"
AMOUNT_COL     = "AmountInBankAccountCurrency"
ACCOUNT_COL    = "BankAccountCode"
CODE_COL       = "BankTransactionCode"
BUSUNIT_COL    = "BusinessUnitCode"
FLAG_CASHBOOK  = "CashBookFlag"
FLAG_CURRDAY   = "IsCurrentDay"   # read; ignore later
IS_MATCHED_COL = "IsMatched"

REQUIRED_COLS = [
    DATE_COL_VALUE, DATE_COL_POST, AMOUNT_COL,
    ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK,
    FLAG_CURRDAY, IS_MATCHED_COL,
]
OPTIONAL_COLS = ["BankTransactionId"]

# Base categoricals (raw)
BASE_CATEGORICAL_COLS = [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]

# ========= Artifacts =========
ART_DIR       = "artifacts_features"
ENCODER_PATH  = os.path.join(ART_DIR, "encoder.pkl")
ENCODER_META  = os.path.join(ART_DIR, "encoder_meta.json")
SCALER_PATH   = os.path.join(ART_DIR, "standard_scaler.pkl")
BASELINE_PATH = os.path.join(ART_DIR, "account_baselines.csv")
SCHEMA_PATH   = os.path.join(ART_DIR, "feature_schema.json")

def _mkdir(p: str): os.makedirs(p, exist_ok=True)

# ========= Config =========
# Numeric (engineered)
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

# Scale these numeric columns (you can change this list)
SCALE_NUM_COLS = ["amount", "mean_amount_30d", "std_amount_30d"]

# Categorical (engineered) – will be fed to the hybrid encoder too
EXTRA_CATEGORICAL_FEATURES = [
    "is_weekend",
    "month",
    "quarter",
    "is_matched_src",
    "cashbook_flag_derived",
]

# Hybrid encoder thresholds
MAX_OHE_CARD = 10         # <= 10 uniques -> one-hot; else -> hashing
HASH_BINS    = 32         # bins per high-card column

# ========= Cleaning & dates =========
def _ensure_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

def _clean_categorical(s: pd.Series) -> pd.Series:
    s = s.astype(object).replace({None: np.nan})
    s = s.astype(str).str.strip()
    blanks = {"", "nan", "NaN", "None", "NULL", "null"}
    return s.map(lambda x: "UNK" if x in blanks else x)

def drop_duplicate_txn_ids(df: pd.DataFrame, col: str = "BankTransactionId", keep: str = "first") -> pd.DataFrame:
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
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {missing}\nFound: {list(df.columns)}")
    if drop_dupes:
        df = drop_duplicate_txn_ids(df, col="BankTransactionId", keep="first")

    for col in [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK, IS_MATCHED_COL]:
        if col in df.columns:
            df[col] = _clean_categorical(df[col])
    return df

# ========= Row features =========
def engineer_row_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount"] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").fillna(0.0)

    val = _ensure_dt(out[DATE_COL_VALUE])
    pst = _ensure_dt(out[DATE_COL_POST])
    out["posting_lag_days"] = (pst - val).dt.days.fillna(0).astype("int16")

    posting = pst.fillna(val)
    out["is_weekend"] = posting.dt.dayofweek.isin([5, 6]).astype("int8")
    out["month"]      = posting.dt.month.astype("int8")
    out["quarter"]    = posting.dt.quarter.astype("int8")

    out["is_matched_src"] = out.get(IS_MATCHED_COL, "UNK").astype(str)
    out["cashbook_flag_derived"] = out[FLAG_CASHBOOK].str.lower().eq("cashbook").astype("int8")

    same_day = posting.dt.date
    grp = out.groupby([ACCOUNT_COL, same_day, "amount"], dropna=False).size()
    out["same_amount_count_per_day"] = grp.loc[
        list(zip(out[ACCOUNT_COL], same_day, out["amount"]))
    ].values.astype("int16")

    out["ts"] = posting
    return out

# ========= Baselines =========
def compute_account_baselines(history_df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(history_df).sort_values([ACCOUNT_COL, "ts"])
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("ts")
        txn_count_7d  = g["amount"].rolling("7D").count().shift(1)
        txn_count_30d = g["amount"].rolling("30D").count().shift(1)
        mean_30d      = g["amount"].rolling("30D").mean().shift(1)
        std_30d       = g["amount"].rolling("30D").std(ddof=1).shift(1)
        avg_lag_30d   = g["posting_lag_days"].rolling("30D").mean().shift(1)
        return (pd.DataFrame({
            "txn_count_7d": txn_count_7d,
            "txn_count_30d": txn_count_30d,
            "mean_amount_30d": mean_30d,
            "std_amount_30d": std_30d,
            "avg_posting_lag_30d": avg_lag_30d,
        }).reset_index())
    base = df.groupby(ACCOUNT_COL, group_keys=True).apply(_roll).reset_index(level=0)
    return base

def merge_with_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(df)
    df["ts"] = pd.to_datetime(df["ts"]); baselines["ts"] = pd.to_datetime(baselines["ts"])

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
    if f"{ACCOUNT_COL}_x" in feats.columns:
        feats.rename(columns={f"{ACCOUNT_COL}_x": ACCOUNT_COL}, inplace=True)
    if f"{ACCOUNT_COL}_y" in feats.columns:
        feats.drop(columns=[f"{ACCOUNT_COL}_y"], inplace=True)

    feats["zscore_amount_30d"] = (
        (feats["amount"] - feats["mean_amount_30d"]) / feats["std_amount_30d"].replace(0, np.nan)
    ).fillna(0.0).astype("float32")

    feats[["txn_count_7d","txn_count_30d"]] = feats[["txn_count_7d","txn_count_30d"]].fillna(0)
    feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]] = (
        feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]].fillna(0.0)
    )
    return feats

# ========= Hybrid Encoder =========
@dataclass
class HybridEncoderArtifacts:
    low_card_cols: List[str]
    high_card_cols: List[str]
    onehot: Any                   # fitted OneHotEncoder over low_card cols (or None)
    max_ohe_card: int
    hash_bins: int

class HybridEncoder:
    """
    Hybrid:
    - One-hot for low-card columns (<= max_ohe_card)
    - Hashing for high-card columns (> max_ohe_card), with per-column HASH_BINS bins
    Deterministic hashing: md5(f"{col}||{token}") % HASH_BINS
    """
    def __init__(self, max_ohe_card: int = MAX_OHE_CARD, hash_bins: int = HASH_BINS):
        self.max_ohe_card = int(max_ohe_card)
        self.hash_bins = int(hash_bins)
        self.low_card_cols: List[str] = []
        self.high_card_cols: List[str] = []
        self.onehot: OneHotEncoder | None = None
        self._fitted = False

    @staticmethod
    def _hash_to_bin(col: str, token: str, bins: int) -> int:
        key = (col + "||" + token).encode("utf-8", errors="ignore")
        # deterministic 128-bit md5 -> int
        h = int(hashlib.md5(key).hexdigest(), 16)
        return h % bins

    def fit(self, df_cat: pd.DataFrame):
        # detect cardinality
        nunique = {c: int(df_cat[c].nunique(dropna=False)) for c in df_cat.columns}
        self.low_card_cols  = [c for c, k in nunique.items() if k <= self.max_ohe_card]
        self.high_card_cols = [c for c, k in nunique.items() if k >  self.max_ohe_card]

        if self.low_card_cols:
            self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot.fit(df_cat[self.low_card_cols])
        else:
            self.onehot = None

        self._fitted = True
        return self

    def transform(self, df_cat: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "HybridEncoder not fitted"

        parts = []
        # low-card one-hot
        if self.onehot is not None and self.low_card_cols:
            X_low = self.onehot.transform(df_cat[self.low_card_cols])
            parts.append(X_low.astype("float32"))

        # high-card hashing (dense one-hot over hash bins per column)
        for col in self.high_card_cols:
            tokens = df_cat[col].astype(str).values
            n = len(tokens)
            mat = np.zeros((n, self.hash_bins), dtype="float32")
            # place 1 in bin for each token
            for i, tok in enumerate(tokens):
                j = self._hash_to_bin(col, tok, self.hash_bins)
                mat[i, j] = 1.0
            parts.append(mat)

        if parts:
            return np.hstack(parts).astype("float32")
        # no categoricals case
        return np.empty((len(df_cat), 0), dtype="float32")

    def get_feature_names_out(self, df_cat_columns: List[str]) -> List[str]:
        names: List[str] = []
        if self.onehot is not None and self.low_card_cols:
            names.extend(self.onehot.get_feature_names_out(self.low_card_cols).tolist())
        for col in self.high_card_cols:
            names.extend([f"{col}__hash_{i}" for i in range(self.hash_bins)])
        return names

    def artifacts(self) -> HybridEncoderArtifacts:
        return HybridEncoderArtifacts(
            low_card_cols=self.low_card_cols,
            high_card_cols=self.high_card_cols,
            onehot=self.onehot,
            max_ohe_card=self.max_ohe_card,
            hash_bins=self.hash_bins,
        )

# ========= Numeric scaling helper =========
def _scale_numeric_block(df_num: pd.DataFrame, scale_cols: List[str]):
    for c in scale_cols:
        if c not in df_num.columns:
            raise ValueError(f"Scale column {c} not in numeric features.")
    if len(scale_cols) == 0:
        return (np.empty((len(df_num), 0), dtype="float32"),
                list(df_num.columns),
                df_num.values.astype("float32"),
                None)
    scaler = StandardScaler().fit(df_num[scale_cols].values)
    scaled_part = scaler.transform(df_num[scale_cols].values).astype("float32")
    passthrough_cols = [c for c in df_num.columns if c not in scale_cols]
    passthrough_part = df_num[passthrough_cols].values.astype("float32")
    return scaled_part, passthrough_cols, passthrough_part, scaler

# ========= Public API =========
@dataclass
class FitArtifacts:
    encoder: HybridEncoder
    scaler: StandardScaler | None
    numeric_cols: List[str]
    categorical_cols: List[str]
    all_feature_cols: List[str]

def fit_featureizer_from_excel(history_xlsx_path: str, sheet_name=0) -> FitArtifacts:
    _mkdir(ART_DIR)
    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)

    base = compute_account_baselines(hist)
    base.to_csv(BASELINE_PATH, index=False)

    feats = merge_with_baselines(hist, base)

    # ---- numeric (configurable scaling) ----
    df_num = feats[NUMERIC_FEATURES].astype("float32").copy()
    scaled_part, passthrough_cols, passthrough_part, scaler = _scale_numeric_block(df_num, SCALE_NUM_COLS)

    # ---- categoricals (hybrid) ----
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)

    hyb = HybridEncoder(max_ohe_card=MAX_OHE_CARD, hash_bins=HASH_BINS).fit(df_cat)
    X_cat = hyb.transform(df_cat)
    cat_names = hyb.get_feature_names_out(df_cat.columns.tolist())

    # ---- final matrix ----
    X = np.hstack([scaled_part, passthrough_part, X_cat]).astype("float32")

    # ---- persist artifacts ----
    with open(ENCODER_PATH, "wb") as f: pickle.dump(hyb, f)
    meta = {
        "type": "hybrid",
        "max_ohe_card": MAX_OHE_CARD,
        "hash_bins": HASH_BINS,
        "low_card_cols": hyb.low_card_cols,
        "high_card_cols": hyb.high_card_cols,
    }
    with open(ENCODER_META, "w") as f: json.dump(meta, f, indent=2)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    schema = {
        "numeric_cols": NUMERIC_FEATURES,
        "scale_numeric_cols": SCALE_NUM_COLS,
        "passthrough_numeric_cols": passthrough_cols,
        "categorical_cols": BASE_CATEGORICAL_COLS + EXTRA_CATEGORICAL_FEATURES,
        "encoded_feature_names": cat_names,
        "hybrid": meta,
    }
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)

    all_feature_cols = [f"{c}_scaled" for c in SCALE_NUM_COLS] + passthrough_cols + cat_names
    return FitArtifacts(
        encoder=hyb,
        scaler=scaler,
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=BASE_CATEGORICAL_COLS + EXTRA_CATEGORICAL_FEATURES,
        all_feature_cols=all_feature_cols,
    )

def transform_excel_batch(batch_xlsx_path: str, sheet_name=0) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    # load artifacts
    with open(ENCODER_PATH, "rb") as f: hyb: HybridEncoder = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)  # may be None

    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    batch = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    feats = merge_with_baselines(batch, base)

    # numeric
    df_num = feats[schema["numeric_cols"]].astype("float32").copy()
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols = schema.get("passthrough_numeric_cols", [c for c in schema["numeric_cols"] if c not in scale_cols])

    if scaler is not None and len(scale_cols) > 0:
        scaled_part = scaler.transform(df_num[scale_cols].values).astype("float32")
    else:
        scaled_part = np.empty((len(df_num), 0), dtype="float32")
    pass_part = df_num[pass_cols].values.astype("float32")

    # categoricals (hybrid)
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)
    X_cat = hyb.transform(df_cat)
    cat_names = hyb.get_feature_names_out(df_cat.columns.tolist())

    # final
    X_final = np.hstack([scaled_part, pass_part, X_cat]).astype("float32")
    feature_names = [f"{c}_scaled" for c in scale_cols] + pass_cols + cat_names
    return X_final, feats, feature_names

def update_baselines_with_excel(batch_xlsx_path: str, sheet_name=0):
    base_old = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    new_df = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base_new = compute_account_baselines(new_df)
    base_all = (pd.concat([base_old, base_new], ignore_index=True)
                 .sort_values([ACCOUNT_COL, "ts"])
                 .drop_duplicates([ACCOUNT_COL, "ts"], keep="last"))
    base_all.to_csv(BASELINE_PATH, index=False)

def build_training_matrix_from_excel(history_xlsx_path: str, sheet_name=0):
    # ensure artifacts
    fit_featureizer_from_excel(history_xlsx_path, sheet_name=sheet_name)

    with open(ENCODER_PATH, "rb") as f: hyb: HybridEncoder = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    feats = merge_with_baselines(hist, base)

    df_num = feats[schema["numeric_cols"]].astype("float32").copy()
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols = schema.get("passthrough_numeric_cols", [c for c in schema["numeric_cols"] if c not in scale_cols])

    if scaler is not None and len(scale_cols) > 0:
        scaled_part = scaler.transform(df_num[scale_cols].values).astype("float32")
    else:
        scaled_part = np.empty((len(df_num), 0), dtype="float32")
    pass_part = df_num[pass_cols].values.astype("float32")

    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)
    X_cat = hyb.transform(df_cat)
    cat_names = hyb.get_feature_names_out(df_cat.columns.tolist())

    X_final = np.hstack([scaled_part, pass_part, X_cat]).astype("float32")
    feature_names = [f"{c}_scaled" for c in scale_cols] + pass_cols + cat_names
    return X_final, feats, feature_names
