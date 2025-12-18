from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# === bring FE (must exist and match training) ===
from preprocessing_and_fe import build_dataset_from_excel
from infer_utils import *
from config import *


class BankAnomalyDetectorWrapper:
    """
    Thin wrapper around the bank anomaly inference pipeline.
    Artifacts are loaded once in __init__ (or injected for tests).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        *,
        artifacts: Optional[Dict[str, Any]] = None,
        predict_batch_size: int = 2048,
    ) -> None:
        if artifacts is None:
            if not model_dir:
                raise ValueError("Provide either model_dir or artifacts.")
            artifacts = load_artifacts(model_dir)

        self.artifacts = artifacts
        self.model = artifacts["model"]
        self.expects_cat_ids = _model_expects_cat_ids(self.model)
        self.predict_batch_size = int(predict_batch_size)

    @staticmethod
    def build_inputs_with_ynorm(
        df: pd.DataFrame,
        tab_cols: List[str],
        y_norm: np.ndarray,
        expects_cat_ids: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        tab = df[tab_cols].astype("float32").values if tab_cols else np.zeros((len(df), 0), dtype="float32")
        y = y_norm.reshape(-1, 1).astype("float32")
        X_in = np.hstack([tab, y]).astype("float32")

        col_names = list(tab_cols) + ["y_norm"]
        j_y = len(col_names) - 1

        if not expects_cat_ids:
            X_aux = df["combo_id"].astype("int32").values
            return X_in, X_aux, col_names, j_y

        combo = df["combo_id"].astype("int32").values
        acct  = df["account_id"].astype("int32").values if "account_id" in df.columns else np.zeros(len(df), dtype="int32")
        bu    = df["bu_id"].astype("int32").values      if "bu_id" in df.columns      else np.zeros(len(df), dtype="int32")
        code  = df["code_id"].astype("int32").values    if "code_id" in df.columns    else np.zeros(len(df), dtype="int32")

        X_aux = np.stack([combo, acct, bu, code], axis=1).astype("int32")
        return X_in, X_aux, col_names, j_y

    def score_bank_statement_df(
        self,
        df_raw: pd.DataFrame,
        *,
        output_path: Optional[str] = None,
        anomaly_col: str = "Anomaly",
        reason_col: str = "AnomalyReason",
    ) -> pd.DataFrame:
        a = self.artifacts  # shorthand

        # base output
        df_out = df_raw.copy().reset_index(drop=True)
        df_out[anomaly_col] = 0
        df_out[reason_col] = ""

        # row mapping safety
        df_fe_in = df_raw.copy().reset_index(drop=True)
        df_fe_in["__row_id__"] = np.arange(len(df_fe_in), dtype=int)

        # 1) FE
        feats_all, _, _ = build_dataset_from_excel(df_fe_in)

        # 2) combo map
        feats_all = apply_combo_map(feats_all, a["combo_map"])

        # 3) cat ids (only if model expects them)
        if self.expects_cat_ids:
            if "account_id" not in feats_all.columns:
                feats_all = _maybe_apply_cat_map(feats_all, "BankAccountCode", a["account_map"], "account_id")
            if "bu_id" not in feats_all.columns:
                feats_all = _maybe_apply_cat_map(feats_all, "BusinessUnitCode", a["bu_map"], "bu_id")
            if "code_id" not in feats_all.columns:
                feats_all = _maybe_apply_cat_map(feats_all, "BankTransactionCode", a["code_map"], "code_id")

            feats_all["account_id"] = feats_all.get("account_id", 0).astype("int32")
            feats_all["bu_id"]      = feats_all.get("bu_id", 0).astype("int32")
            feats_all["code_id"]    = feats_all.get("code_id", 0).astype("int32")

        tab_cols_train = a["tab_cols_train"]

        # 4) count_norm (if part of training)
        if "count_norm" in tab_cols_train:
            feats_all = add_count_norm(
                feats_all,
                a["stats_train"],
                count_day_col=a["count_day_col"],
                count_iqr_floor=a["count_iqr_floor"],
                clip_lo=a["count_norm_clip_lo"],
                clip_hi=a["count_norm_clip_hi"],
            )

        # 5) align columns
        for col in tab_cols_train:
            if col not in feats_all.columns:
                feats_all[col] = 0.0
        feats_for_model = feats_all.copy()
        feats_for_model[tab_cols_train] = feats_for_model[tab_cols_train].astype("float32")

        # 6) y_norm
        y_norm = normalize_amount(feats_for_model, a["stats_train"])

        # 7) inputs + weights
        X_in, X_aux, col_names, j_y = self.build_inputs_with_ynorm(
            feats_for_model, tab_cols_train, y_norm, expects_cat_ids=self.expects_cat_ids
        )

        col_w = build_col_weights(
            tab_cols=tab_cols_train,
            j_y=j_y,
            weight_y_norm=a["weight_y_norm"],
            weight_tabular=a["weight_tabular"],
            weight_count_norm=a["weight_count_norm"],
        )

        # 8) predict + recon error
        X_pred = self.model.predict([X_in, X_aux], batch_size=self.predict_batch_size, verbose=0)
        err = row_recon_error(X_in, X_pred, col_w)

        # optional count prediction
        if "count_norm" in tab_cols_train:
            j_cnt = col_names.index("count_norm")
            cnt_pred_float = invert_count_norm_to_count(
                feats_for_model,
                X_pred[:, j_cnt],
                a["stats_train"],
                count_iqr_floor=a["count_iqr_floor"],
            )
            feats_for_model["trans_count_day_pred"] = np.rint(cnt_pred_float).astype(int)

        combo_arr = feats_for_model["combo_str"].astype(str).values
        thr_vec = np.array([a["thr_per_combo"].get(c, a["thr_global"]) for c in combo_arr], dtype=float)
        mask_recon = err >= thr_vec

        # 9) INR tolerance
        amt_pred = invert_pred_to_inr(feats_for_model, X_pred[:, j_y], a["stats_train"])
        amt_act  = feats_for_model["amount"].astype(float).values

        mad_vec = np.array([a["stats_train"].get(c, {}).get("mad_inr", 1.0) for c in combo_arr], dtype=float)
        tol_abs = np.maximum.reduce([
            np.full_like(amt_act, a["abs_tol_inr"], dtype=float),
            a["mad_mult"] * mad_vec,
            a["pct_tol"] * np.abs(amt_act),
        ])

        diff_abs = np.abs(amt_pred - amt_act)
        mask_inr = diff_abs >= tol_abs

        is_anom = mask_recon & mask_inr
        idx = np.where(is_anom)[0]

        if len(idx) > 0:
            out = feats_for_model.iloc[idx].copy()
            out["amount_pred"] = amt_pred[idx]
            out["amount_diff_abs"] = diff_abs[idx]
            out["recon_error"] = err[idx]
            out["thr_recon"] = thr_vec[idx]

            dev = compute_top_feature_deviations(
                X_true=X_in,
                X_pred=X_pred,
                col_names=col_names,
                j_y=j_y,
                anomaly_idx=idx,
                col_w=col_w,
                top_k=2,
            )
            for k, vals in dev.items():
                out[k] = vals

            out["reason"] = out.apply(format_reason, axis=1)

            row_ids = out["__row_id__"].astype(int).values
            df_out.loc[row_ids, anomaly_col] = 1
            df_out.loc[row_ids, reason_col] = out["reason"].values

        if output_path is not None:
            import os
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df_out.to_csv(output_path, index=False)

        return df_out

