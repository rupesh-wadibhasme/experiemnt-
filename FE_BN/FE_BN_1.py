# bank_features_mvp_xls.py
# MVP feature engineering for unsupervised anomaly detection (autoencoder)
# Reads ONLY the required columns from Excel using `usecols=`.

from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# Column map (updated)
# =========================
DATE_COL_VALUE = "ValueDateKey"              # int/str like 20230102
DATE_COL_POST  = "PostingDateKey"            # int/str like 20230102
AMOUNT_COL     = "AmountInBankAccountCurrency"
ACCOUNT_COL    = "BankAccountCode"
CODE_COL       = "BankTransactionCode"
BUSUNIT_COL    = "BusinessUnitCode"
FLAG_CASHBOOK  = "CashBookFlag"              # e.g., 'Cashbook' / 'NonCashbook'
FLAG_CURRDAY   = "IsCurrentDay"              # 0/1
MATCHED_REF    = "MatchedReference"          # 0/blank/ID

# Columns required for feature engineering / encoding
REQUIRED_COLS = [
    DATE_COL_VALUE, DATE_COL_POST, AMOUNT_COL,
    ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK,
    FLAG_CURRDAY, MATCHED_REF,
]
# Optional passthrough IDs we keep if present (not used for features)
OPTIONAL_COLS = ["BankTransactionId"]

CATEGORICAL_COLS = [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]

# =========================
# Artifacts
# =========================
ART_DIR       = "artifacts_features"
ENCODER_PATH  = os.path.join(ART_DIR, "onehot_encoder.pkl")
SCALER_PATH   = os.path.join(ART_DIR, "standard_scaler.pkl")
BASELINE_PATH = os.path.join(ART_DIR, "account_baselines.csv")   # CSV to avoid parquet deps
SCHEMA_PATH   = os.path.join(ART_DIR, "feature_schema.json")

# =========================
# Helpers
# =========================
def _mkdir(p: str): os.makedirs(p, exist_ok=True)

def _ensure_dt(series: pd.Series) -> pd.Series:
    """Parse YYYYMMDD ints/strings -> datetime64[ns]."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

def _amount_sign(x: float) -> int:
    return int((x > 0) - (x < 0))

def _read_excel_selected(path: str, sheet_name=0) -> pd.DataFrame:
    """
    Read ONLY the required + optional columns from Excel.
    If a required column is missing, raises a clear error.
    """
    usecols = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c not in REQUIRED_COLS]
    # Only set dtype for the date keys so pandas doesn't guess formats
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
    return df

# =========================
# Row-level features (no history)
# =========================
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
    out["has_matched_ref"] = (~out.get(MATCHED_REF, pd.Series(index=out.index))
                                .astype(str).str.strip().isin(["", "0", "nan", "NaN"])).astype("int8")
    out["cashbook_flag"]    = out.get(FLAG_CASHBOOK, "").astype(str).str.lower().eq("cashbook").astype("int8")
    out["current_day_flag"] = pd.to_numeric(out.get(FLAG_CURRDAY, 0), errors="coerce").fillna(0).astype("int8")

    # Same-amount count (same account, same posting date)
    same_day = posting.dt.date
    grp = out.groupby([ACCOUNT_COL, same_day, "amount"], dropna=False).size()
    out["same_amount_count_per_day"] = grp.loc[list(zip(out[ACCOUNT_COL], same_day, out["amount"]))].values.astype("int16")

    out["ts"] = posting  # canonical timestamp for joins
    return out

# =========================
# Baselines: per-account rolling stats (no leakage)
# =========================
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

    # Keep account column in the baseline table
    base = df.groupby(ACCOUNT_COL, group_keys=True).apply(_roll).reset_index(level=0).rename(columns={ACCOUNT_COL: ACCOUNT_COL})
    return base

def merge_with_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    df = engineer_row_level(df)

    # Ensure datetime
    df["ts"] = pd.to_datetime(df["ts"])
    baselines["ts"] = pd.to_datetime(baselines["ts"])

    # Per-account asof merge
    out_parts = []
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
        out_parts.append(merged)

    feats = pd.concat(out_parts, ignore_index=True)
    # Normalize account columns if suffixes appear
    if f"{ACCOUNT_COL}_x" in feats.columns:
        feats.rename(columns={f"{ACCOUNT_COL}_x": ACCOUNT_COL}, inplace=True)
    if f"{ACCOUNT_COL}_y" in feats.columns:
        feats.drop(columns=[f"{ACCOUNT_COL}_y"], inplace=True)

    # z-score vs 30d history
    feats["zscore_amount_30d"] = (feats["amount"] - feats["mean_amount_30d"]) / feats["std_amount_30d"].replace(0, np.nan)
    feats["zscore_amount_30d"] = feats["zscore_amount_30d"].fillna(0.0).astype("float32")

    # Fill cold starts
    feats[["txn_count_7d","txn_count_30d"]] = feats[["txn_count_7d","txn_count_30d"]].fillna(0)
    feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]] = feats[["mean_amount_30d","std_amount_30d","avg_posting_lag_30d"]].fillna(0.0)
    return feats

# =========================
# Fit / Transform API
# =========================
@dataclass
class FitArtifacts:
    encoder: OneHotEncoder
    scaler: StandardScaler
    numeric_cols: List[str]
    categorical_cols: List[str]
    all_feature_cols: List[str]

def fit_featureizer_from_excel(history_xlsx_path: str, sheet_name=0) -> FitArtifacts:
    _mkdir(ART_DIR)
    hist = _read_excel_selected(history_xlsx_path, sheet_name=sheet_name)

    base = compute_account_baselines(hist)
    base.to_csv(BASELINE_PATH, index=False)

    feats = merge_with_baselines(hist, base)

    numeric_cols = [
        "amount","amount_log","amount_sign","posting_lag_days","same_amount_count_per_day",
        "txn_count_7d","txn_count_30d","mean_amount_30d","std_amount_30d","avg_posting_lag_30d",
        "zscore_amount_30d","day_of_week","is_weekend","day_of_month","month","quarter",
        "has_matched_ref","cashbook_flag","current_day_flag",
    ]
    categorical_cols = [ACCOUNT_COL, CODE_COL, BUSUNIT_COL, FLAG_CASHBOOK]

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(feats[categorical_cols])

    X_num = feats[numeric_cols].astype("float32").values
    X_cat = enc.transform(feats[categorical_cols])
    X = np.hstack([X_num, X_cat]).astype("float32")

    scaler = StandardScaler().fit(X)

    # Persist artifacts
    with open(ENCODER_PATH, "wb") as f: pickle.dump(enc, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    schema = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ohe_feature_names": enc.get_feature_names_out(categorical_cols).tolist()
    }
    with open(SCHEMA_PATH, "w") as f: json.dump(schema, f, indent=2)

    return FitArtifacts(
        encoder=enc,
        scaler=scaler,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        all_feature_cols=numeric_cols + schema["ohe_feature_names"]
    )

def transform_excel_batch(batch_xlsx_path: str, sheet_name=0) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    # Load artifacts
    with open(ENCODER_PATH, "rb") as f: enc: OneHotEncoder = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler: StandardScaler = pickle.load(f)
    with open(SCHEMA_PATH, "r") as f: schema = json.load(f)
    base = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])

    # Read ONLY the needed columns from the batch
    batch = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name)
    feats = merge_with_baselines(batch, base)

    # Build matrix
    X_num = feats[schema["numeric_cols"]].astype("float32").values
    X_cat = enc.transform(feats[schema["categorical_cols"]])
    X = np.hstack([X_num, X_cat]).astype("float32")
    X_scaled = scaler.transform(X)

    feat_names = schema["numeric_cols"] + enc.get_feature_names_out(schema["categorical_cols"]).tolist()
    return X_scaled, feats, feat_names

def update_baselines_with_excel(batch_xlsx_path: str, sheet_name=0):
    """Call this after a batch is successfully scored to keep baselines fresh."""
    base_old = pd.read_csv(BASELINE_PATH, parse_dates=["ts"])
    new_df = _read_excel_selected(batch_xlsx_path, sheet_name=sheet_name)
    base_new = compute_account_baselines(new_df)

    base_all = (pd.concat([base_old, base_new], ignore_index=True)
                  .sort_values([ACCOUNT_COL, "ts"])
                  .drop_duplicates([ACCOUNT_COL, "ts"], keep="last"))
    base_all.to_csv(BASELINE_PATH, index=False)
