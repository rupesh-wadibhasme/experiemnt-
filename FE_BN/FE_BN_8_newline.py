# bank_features_training_xls.py
# Feature engineering for unsupervised anomaly detection.
# - Numeric features (configurable subset; selected ones scaled with StandardScaler)
# - Categorical features (ALL of them) encoded with category_encoders.BinaryEncoder
# - Persist artifacts: scaler, binary encoder, schema, and rolling baselines (no leakage)

from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder

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

# Key for combo conditioning
GROUP_KEYS = [ACCOUNT_COL, BUSUNIT_COL, CODE_COL]

# ========= Artifacts =========
ART_DIR       = "artifacts_features"
ENCODER_PATH  = os.path.join(ART_DIR, "encoder.pkl")           # BinaryEncoder
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
    # cyclical time features (aid reconstruction)
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    # optional stability feature
    "log_amount",
]

# Scale these numeric columns
SCALE_NUM_COLS = ["amount", "mean_amount_30d", "std_amount_30d", "log_amount"]

# Categorical (engineered) – encoded via BinaryEncoder
EXTRA_CATEGORICAL_FEATURES = [
    "is_weekend",
    "month",
    "quarter",
    "is_matched_src",
    "cashbook_flag_derived",
]

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
    
def _assert_has_columns(df: pd.DataFrame, cols: list, where: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {where or 'DataFrame'}. "
                       f"Available: {list(df.columns)}")

# ========= Helpers =========
def _map_cashbook(s: pd.Series) -> pd.Series:
    """Safer mapping for cashbook-like flags."""
    s = s.astype(str).str.strip().str.lower()
    return s.isin({"cashbook", "y", "yes", "1", "true"}).astype("int8")

# ========= Row features =========
def engineer_row_level(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level numeric + calendar features used later in rolling baselines."""
    out = df.copy()
    out["amount"] = pd.to_numeric(out[AMOUNT_COL], errors="coerce").fillna(0.0)

    val = _ensure_dt(out[DATE_COL_VALUE])
    pst = _ensure_dt(out[DATE_COL_POST])

    # clamp posting lag to avoid wild outliers due to data errors
    lag = (pst - val).dt.days
    lag = lag.fillna(0).clip(lower=-182, upper=182)  # ± ~6 months
    out["posting_lag_days"] = lag.astype("int16")

    posting = pst.fillna(val)
    out["is_weekend"] = posting.dt.dayofweek.isin([5, 6]).astype("int8")
    out["month"]      = posting.dt.month.astype("int8")
    out["quarter"]    = posting.dt.quarter.astype("int8")

    # cyclical encodings for better reconstruction
    out["month_sin"]  = np.sin(2*np.pi*(out["month"].astype(float) / 12.0)).astype("float32")
    out["month_cos"]  = np.cos(2*np.pi*(out["month"].astype(float) / 12.0)).astype("float32")
    out["dow"]        = posting.dt.dayofweek.astype("int8")
    out["dow_sin"]    = np.sin(2*np.pi*(out["dow"].astype(float) / 7.0)).astype("float32")
    out["dow_cos"]    = np.cos(2*np.pi*(out["dow"].astype(float) / 7.0)).astype("float32")

    out["is_matched_src"] = out.get(IS_MATCHED_COL, "UNK").astype(str)
    out["cashbook_flag_derived"] = _map_cashbook(out[FLAG_CASHBOOK])

    # Align same-amount-per-day to the combo keys
    same_day = posting.dt.date
    grp = out.groupby(GROUP_KEYS + [same_day, "amount"], dropna=False).size()
    out["same_amount_count_per_day"] = grp.loc[
        list(zip(out[ACCOUNT_COL], out[BUSUNIT_COL], out[CODE_COL], same_day, out["amount"]))
    ].values.astype("int16")

    # optional heavy-tail stabilizer for training
    out["log_amount"] = (np.log1p(out["amount"].abs()) * np.sign(out["amount"])).astype("float32")

    # identifiers helpful for downstream diagnostics (not encoded)
    out["combo_id"] = (out[ACCOUNT_COL].astype(str) + "|" +
                       out[BUSUNIT_COL].astype(str) + "|" +
                       out[CODE_COL].astype(str))

    out["ts"] = posting
    return out

# ========= Baselines (per Account, BU, Code) =========
def compute_account_baselines(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-free rolling 7D/30D stats PER (Account, BU, Code).
    Uses shift(1) so the current row/day is not included in its own baseline.
    """
    df = engineer_row_level(history_df).copy()
    _assert_has_columns(df, GROUP_KEYS + ["ts", "amount", "posting_lag_days"], "compute_account_baselines.input")

    # sort by all keys + time
    df = df.sort_values(GROUP_KEYS + ["ts"]).reset_index(drop=True)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        # capture key values BEFORE reindex to avoid surprises
        key_vals = {k: g.iloc[0][k] for k in GROUP_KEYS}
        g2 = g.set_index("ts")

        txn_count_7d  = g2["amount"].rolling("7D").count().shift(1)
        txn_count_30d = g2["amount"].rolling("30D").count().shift(1)
        mean_30d      = g2["amount"].rolling("30D").mean().shift(1)
        std_30d       = g2["amount"].rolling("30D").std(ddof=1).shift(1)
        avg_lag_30d   = g2["posting_lag_days"].rolling("30D").mean().shift(1)

        out = (pd.DataFrame({
            "txn_count_7d": txn_count_7d,
            "txn_count_30d": txn_count_30d,
            "mean_amount_30d": mean_30d,
            "std_amount_30d": std_30d,
            "avg_posting_lag_30d": avg_lag_30d,
        }).reset_index())  # brings 'ts' back

        for k, v in key_vals.items():
            out[k] = v
        return out

    base = (df.groupby(GROUP_KEYS, dropna=False, sort=False)
              .apply(_roll)
              .reset_index(drop=True))

    _assert_has_columns(base, GROUP_KEYS + ["ts"], "compute_account_baselines.output")
    base = base.sort_values(GROUP_KEYS + ["ts"]).reset_index(drop=True)
    return base


def merge_with_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    As-of merge leakage-free rolling stats PER (Account, BU, Code).
    For each exact combo in GROUP_KEYS, asof-join to the most recent baseline <= row.ts.
    """
    feat = engineer_row_level(df).copy()
    _assert_has_columns(feat, GROUP_KEYS + ["ts", "amount"], "merge_with_baselines.input.feat")

    feat["ts"] = pd.to_datetime(feat["ts"])
    base = baselines.copy()
    _assert_has_columns(base, GROUP_KEYS + ["ts"], "merge_with_baselines.input.base")
    base["ts"] = pd.to_datetime(base["ts"])

    parts = []
    for keys_vals, g in feat.groupby(GROUP_KEYS, dropna=False, sort=False):
        left = g.sort_values("ts")
        if not isinstance(keys_vals, tuple):
            keys_vals = (keys_vals,)
        # exact-match slice of baseline
        mask = pd.Series(True, index=base.index)
        for k, v in zip(GROUP_KEYS, keys_vals):
            mask &= (base[k] == v)
        right = base.loc[mask].sort_values("ts")

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

    # Robust z-score vs the combo baseline
    eps = 1e-6
    std = feats["std_amount_30d"].copy()
    zero_std = std <= eps
    std_adj = std.mask(zero_std, np.nan)

    z = (feats["amount"] - feats["mean_amount_30d"]) / std_adj
    # if std==0: z=0 when amount==mean, else spike (e.g., 6.0)
    z = z.where(~zero_std, np.where(
        (feats["amount"] - feats["mean_amount_30d"]).abs() <= eps,
        0.0,
        6.0
    ))
    feats["zscore_amount_30d"] = z.fillna(0.0).astype("float32")

    feats[["txn_count_7d","txn_count_30d"]] = feats[["txn_count_7d","txn_count_30d"]].fillna(0)
    feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]] = (
        feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]].fillna(0.0)
    )
    return feats



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
    encoder: Any                  # BinaryEncoder
    scaler: StandardScaler | None
    numeric_cols: List[str]
    categorical_cols: List[str]
    all_feature_cols: List[str]

def _fit_binary_encoder(df_cat: pd.DataFrame) -> Tuple[BinaryEncoder, pd.DataFrame, List[str]]:
    """
    Fit BinaryEncoder on categorical frame and return (encoder, transformed_df, encoded_col_names).
    BinaryEncoder yields compact binary columns per category value (scales to high-cardinality).
    """
    enc = BinaryEncoder(
        cols=df_cat.columns.tolist(),
        drop_invariant=True,
        handle_missing='value',
        handle_unknown='value',
        return_df=True
    )
    X_cat_df = enc.fit_transform(df_cat.astype(str))
    cat_names = X_cat_df.columns.tolist()
    return enc, X_cat_df, cat_names

def fit_featureizer_from_excel(history_xlsx_path: str, sheet_name=0) -> FitArtifacts:
    """
    Fit full featureizer from an Excel history:
      1) Compute leakage-free rolling baselines and merge.
      2) Fit scaler on selected numeric cols; fit BinaryEncoder on all categorical cols.
      3) Persist artifacts and return column metadata.
    """
    _mkdir(ART_DIR)
    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)

    base = compute_account_baselines(hist)
    base.to_csv(BASELINE_PATH, index=False)

    feats = merge_with_baselines(hist, base)

    # ---- numeric (configurable scaling) ----
    df_num = feats[NUMERIC_FEATURES].astype("float32").copy()
    scaled_part, passthrough_cols, passthrough_part, scaler = _scale_numeric_block(df_num, SCALE_NUM_COLS)

    # ---- categoricals (BinaryEncoder over ALL categoricals) ----
    df_cat = feats[BASE_CATEGORICAL_COLS].copy()
    for col in EXTRA_CATEGORICAL_FEATURES:
        df_cat[col] = feats[col].astype(str)
    enc, X_cat_df, cat_names = _fit_binary_encoder(df_cat)

    # ---- final matrix ----
    X = np.hstack([scaled_part, passthrough_part, X_cat_df.values]).astype("float32")

    # ---- persist artifacts ----
    with open(ENCODER_PATH, "wb") as f: pickle.dump(enc, f)
    meta = {
        "type": "binary",
        "encoder": "category_encoders.BinaryEncoder",
        "drop_invariant": True,
        "handle_missing": "value",
        "handle_unknown": "value",
    }
    with open(ENCODER_META, "w") as f: json.dump(meta, f, indent=2)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    schema = {
        "numeric_cols": NUMERIC_FEATURES,
        "scale_numeric_cols": SCALE_NUM_COLS,
        "passthrough_numeric_cols": passthrough_cols,
        "categorical_cols": BASE_CATEGORICAL_FEATURES + EXTRA_CATEGORICAL_FEATURES,
        "encoded_feature_names": cat_names,
        "binary": meta,
    }
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)

    all_feature_cols = [f"{c}_scaled" for c in SCALE_NUM_COLS] + passthrough_cols + cat_names
    return FitArtifacts(
        encoder=enc,
        scaler=scaler,
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=BASE_CATEGORICAL_FEATURES + EXTRA_CATEGORICAL_FEATURES,
        all_feature_cols=all_feature_cols,
    )

def _transform_with_artifacts(feats: pd.DataFrame,
                              enc: BinaryEncoder,
                              scaler: StandardScaler | None,
                              schema: Dict) -> Tuple[np.ndarray, List[str]]:
    """Apply saved scaler + BinaryEncoder to a new engineered frame and return (X, feature_names)."""
    # numeric
    df_num = feats[schema["numeric_cols"]].astype("float32").copy()
    scale_cols = schema.get("scale_numeric_cols", [])
    pass_cols = schema.get("passthrough_numeric_cols", [c for c in schema["numeric_cols"] if c not in scale_cols])

    if scaler is not None and len(scale_cols) > 0:
        scaled_part = scaler.transform(df_num[scale_cols].values).astype("float32")
    else:
        scaled_part = np.empty((len(df_num), 0), dtype="float32")
    pass_part = df_num[pass_cols].values.astype("float32")

    # categoricals
    df_cat = feats[[c for c in (BASE_CATEGORICAL_COLS + EXTRA_CATEGORICAL_FEATURES) if c in feats.columns]].copy()
    for c in df_cat.columns:
        df_cat[c] = df_cat[c].astype(str)
    X_cat_df = enc.transform(df_cat)
    cat_names = X_cat_df.columns.tolist()

    X_final = np.hstack([scaled_part, pass_part, X_cat_df.values]).astype("float32")
    feature_names = [f"{c}_scaled" for c in scale_cols] + pass_cols + cat_names
    return X_final, feature_names

def transform_excel_batch(batch_xlsx_path: str, sheet_name=0) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Transform a fresh Excel batch using saved artifacts:
      - Reads raw, merges leakage-free baselines, then applies scaler + BinaryEncoder.
      - Returns (X, engineered_features_df, feature_names).
    """
    # load artifacts
    with open(ENCODER_PATH, "rb") as f: enc: BinaryEncoder = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)  # may be None

    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    batch = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    feats = merge_with_baselines(batch, base)

    X_final, feature_names = _transform_with_artifacts(feats, enc, scaler, schema)
    return X_final, feats, feature_names

def update_baselines_with_excel(batch_xlsx_path: str, sheet_name=0):
    """
    Recompute baselines on the incoming batch and merge with the existing baseline store,
    keeping the latest record per (GROUP_KEYS, ts).
    """
    base_old = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    _assert_has_columns(base_old, GROUP_KEYS + ["ts"], "update_baselines_with_excel.base_old")

    new_df = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base_new = compute_account_baselines(new_df)

    base_all = (pd.concat([base_old, base_new], ignore_index=True)
                  .sort_values(GROUP_KEYS + ["ts"])
                  .drop_duplicates(GROUP_KEYS + ["ts"], keep="last"))
    base_all.to_csv(BASELINE_PATH, index=False)

def build_training_matrix_from_excel(history_xlsx_path: str, sheet_name=0):
    """
    One-shot builder for training:
      - Ensures artifacts are fitted/saved, then produces (X, engineered_df, feature_names) for the history.
    """
    # ensure artifacts
    fit_featureizer_from_excel(history_xlsx_path, sheet_name=sheet_name)

    with open(ENCODER_PATH, "rb") as f: enc: BinaryEncoder = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name, drop_dupes=True)
    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    feats = merge_with_baselines(hist, base)

    X_final, feature_names = _transform_with_artifacts(feats, enc, scaler, schema)
    return X_final, feats, feature_names
