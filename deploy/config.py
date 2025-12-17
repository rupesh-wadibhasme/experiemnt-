from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise ImportError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e


_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_yaml(path: Path = _CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root in {path}. Expected a dict.")
    return cfg


CFG: Dict[str, Any] = _load_yaml()

# -------------------------
# Preprocessing config
# -------------------------
ART_DIR = str(CFG["preprocessing"]["art_dir"])

COMBO_MAPPING_JSON = os.path.join(ART_DIR, str(CFG["preprocessing"]["files"]["combo_mapping_json"]))
SCHEMA_JSON = os.path.join(ART_DIR, str(CFG["preprocessing"]["files"]["schema_json"]))

ACCOUNT_MAP_JSON = os.path.join(ART_DIR, str(CFG["preprocessing"]["files"]["account_map_json"]))
BUSUNIT_MAP_JSON = os.path.join(ART_DIR, str(CFG["preprocessing"]["files"]["busunit_map_json"]))
CODE_MAP_JSON = os.path.join(ART_DIR, str(CFG["preprocessing"]["files"]["code_map_json"]))

OOV_TOKEN = str(CFG["preprocessing"]["oov_token"])

DATE_COL_VALUE = str(CFG["preprocessing"]["columns"]["date_col_value"])
DATE_COL_POST = str(CFG["preprocessing"]["columns"]["date_col_post"])
AMOUNT_COL = str(CFG["preprocessing"]["columns"]["amount_col"])
ACCOUNT_COL = str(CFG["preprocessing"]["columns"]["account_col"])
CODE_COL = str(CFG["preprocessing"]["columns"]["code_col"])
BUSUNIT_COL = str(CFG["preprocessing"]["columns"]["busunit_col"])
FLAG_CASHBOOK = str(CFG["preprocessing"]["columns"]["flag_cashbook"])
TXN_ID_COL = str(CFG["preprocessing"]["columns"]["txn_id_col"])

REQUIRED_COLS: List[str] = list(CFG["preprocessing"]["columns"]["required_cols"])

EPS_MEAN = float(CFG["preprocessing"]["numerics"]["eps_mean"])


# -------------------------
# Training config
# -------------------------
OUT_DIR = str(CFG["training"]["out_dir"])

OUTPUT_CSV = str(CFG["training"]["artifacts"]["output_csv"])
LEARNING_CURVE_PNG = str(CFG["training"]["artifacts"]["learning_curve_png"])
META_JSON = str(CFG["training"]["artifacts"]["meta_json"])
MODEL_PATH = str(CFG["training"]["artifacts"]["model_path"])
COMBO_MAP_JSON = str(CFG["training"]["artifacts"]["combo_map_json"])
COMBO_STATS_JSON = str(CFG["training"]["artifacts"]["combo_stats_json"])
THRESHOLDS_JSON = str(CFG["training"]["artifacts"]["thresholds_json"])

VALID_FRAC_IN_TRAIN = float(CFG["training"]["split"]["valid_frac_in_train"])

EMBED_DIM = int(CFG["training"]["model"]["embed_dim"])
ENC_UNITS = tuple(int(x) for x in CFG["training"]["model"]["enc_units"])
DEC_UNITS = tuple(int(x) for x in CFG["training"]["model"]["dec_units"])
LR = float(CFG["training"]["model"]["lr"])
BATCH_SIZE = int(CFG["training"]["model"]["batch_size"])
EPOCHS = int(CFG["training"]["model"]["epochs"])
PATIENCE = int(CFG["training"]["model"]["patience"])

WEIGHT_YNORM = float(CFG["training"]["loss_weights"]["weight_ynorm"])
WEIGHT_TABULAR = float(CFG["training"]["loss_weights"]["weight_tabular"])

THRESHOLD_PERCENTILE = float(CFG["training"]["thresholding"]["threshold_percentile"])
MIN_SAMPLES_PER_COMBO_THR = int(CFG["training"]["thresholding"]["min_samples_per_combo_thr"])

ABS_TOL_INR = float(CFG["training"]["tolerance"]["abs_tol_inr"])
PCT_TOL = float(CFG["training"]["tolerance"]["pct_tol"])
MAD_MULT = float(CFG["training"]["tolerance"]["mad_mult"])

GLOBAL_SEED = int(CFG["training"]["reproducibility"]["global_seed"])

COUNT_IQR_FLOOR = float(CFG["training"]["count_normalization"]["count_iqr_floor"])
COUNT_NORM_CLIP_LO = float(CFG["training"]["count_normalization"]["count_norm_clip_lo"])
COUNT_NORM_CLIP_HI = float(CFG["training"]["count_normalization"]["count_norm_clip_hi"])
COUNT_DAY_COL = str(CFG["training"]["count_normalization"]["count_day_col"])

IGNORE_FOR_RECON = set(CFG["training"]["features"]["ignore_for_recon"])

# OOV combo id tag (must match training) — alias to keep your naming as-is
OOV_ID_NAME = OOV_TOKEN


# -------------------------
# Inference config (unique)
# -------------------------
ANOMALY_COL_NAME = str(CFG["inference"]["columns"]["anomaly_col_name"])
REASON_COL_NAME = str(CFG["inference"]["columns"]["reason_col_name"])

ACCOUNT_MAP_JSON_NAME = str(CFG["inference"]["optional_map_names"]["account_map_json_name"])
BU_MAP_JSON_NAME = str(CFG["inference"]["optional_map_names"]["bu_map_json_name"])
CODE_MAP_JSON_NAME = str(CFG["inference"]["optional_map_names"]["code_map_json_name"])

# Duplicated inference aliases (requested names)
OUT_DIR_DEFAULT = OUT_DIR

MODEL_PATH_NAME = MODEL_PATH
META_JSON_NAME = META_JSON
COMBO_MAP_JSON_NAME = COMBO_MAP_JSON
COMBO_STATS_JSON_NAME = COMBO_STATS_JSON
THRESHOLDS_JSON_NAME = THRESHOLDS_JSON


# -------------------------
# Optional convenience: full output paths under OUT_DIR
# (Does not affect your existing names; just helpful)
# -------------------------
OUTPUT_CSV_PATH = os.path.join(OUT_DIR, OUTPUT_CSV)
LEARNING_CURVE_PNG_PATH = os.path.join(OUT_DIR, LEARNING_CURVE_PNG)
META_JSON_PATH = os.path.join(OUT_DIR, META_JSON)
MODEL_PATH_FULL = os.path.join(OUT_DIR, MODEL_PATH)
COMBO_MAP_JSON_PATH = os.path.join(OUT_DIR, COMBO_MAP_JSON)
COMBO_STATS_JSON_PATH = os.path.join(OUT_DIR, COMBO_STATS_JSON)
THRESHOLDS_JSON_PATH = os.path.join(OUT_DIR, THRESHOLDS_JSON)


# -------------------------
# Wildcard import control
# -------------------------
__all__ = [
    # preprocessing
    "ART_DIR",
    "COMBO_MAPPING_JSON", "SCHEMA_JSON",
    "ACCOUNT_MAP_JSON", "BUSUNIT_MAP_JSON", "CODE_MAP_JSON",
    "OOV_TOKEN",
    "DATE_COL_VALUE", "DATE_COL_POST", "AMOUNT_COL",
    "ACCOUNT_COL", "CODE_COL", "BUSUNIT_COL", "FLAG_CASHBOOK",
    "TXN_ID_COL", "REQUIRED_COLS",
    "EPS_MEAN",

    # training
    "OUT_DIR",
    "OUTPUT_CSV", "LEARNING_CURVE_PNG",
    "META_JSON", "MODEL_PATH",
    "COMBO_MAP_JSON", "COMBO_STATS_JSON", "THRESHOLDS_JSON",
    "VALID_FRAC_IN_TRAIN",
    "EMBED_DIM", "ENC_UNITS", "DEC_UNITS", "LR",
    "BATCH_SIZE", "EPOCHS", "PATIENCE",
    "WEIGHT_YNORM", "WEIGHT_TABULAR",
    "THRESHOLD_PERCENTILE", "MIN_SAMPLES_PER_COMBO_THR",
    "ABS_TOL_INR", "PCT_TOL", "MAD_MULT",
    "GLOBAL_SEED",
    "COUNT_IQR_FLOOR", "COUNT_NORM_CLIP_LO", "COUNT_NORM_CLIP_HI", "COUNT_DAY_COL",
    "OOV_ID_NAME",
    "IGNORE_FOR_RECON",

    # inference
    "OUT_DIR_DEFAULT",
    "MODEL_PATH_NAME", "META_JSON_NAME",
    "COMBO_MAP_JSON_NAME", "COMBO_STATS_JSON_NAME", "THRESHOLDS_JSON_NAME",
    "ACCOUNT_MAP_JSON_NAME", "BU_MAP_JSON_NAME", "CODE_MAP_JSON_NAME",
    "ANOMALY_COL_NAME", "REASON_COL_NAME",

    # convenience paths (optional)
    "OUTPUT_CSV_PATH", "LEARNING_CURVE_PNG_PATH",
    "META_JSON_PATH", "MODEL_PATH_FULL",
    "COMBO_MAP_JSON_PATH", "COMBO_STATS_JSON_PATH", "THRESHOLDS_JSON_PATH",
]

