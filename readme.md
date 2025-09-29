
# FX Anomaly Detection — Databricks MLflow Serving

A modular inference stack for detecting anomalous FX deals using a multi‑input Keras autoencoder served on Databricks Model Serving. Training runs in seperate notebook and produced artifacts consumed here at inference time. This repository focuses on inference, serving, security, and operability.

> `infer.py` is the main entry point used for inference (MLflow `pyfunc` model).  
> `env-config.yml`, `system_prompt.json`, and `log_utils.py` are intentionally omitted from source control; see **Configuration** and **Observability** for their roles.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Artifacts Produced by Training](#artifacts-produced-by-training)
- [Inference API](#inference-api)
- [Deployment — Databricks Model Serving](#deployment--databricks-model-serving)
- [Observability](#observability)
- [Security Hardening](#security-hardening)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Modular inference**: clean separation between artifact IO, preprocessing, business rules, model scoring, and serialization.
- **Business-logic short‑circuits**: explicit anomaly reasons without engaging the model when rules trigger (e.g., first‑time pairs).
- **Multi‑input autoencoder**: reconstructs categorical (as one‑hot) and numeric inputs; deviations drive anomaly decisions.
- **Secure deserialization**: restricted unpickling to prevent arbitrary code execution when loading scaler artifacts.
- **Databricks‑native**: built for MLflow Model Registry and Model Serving. Uses AAD application credentials for SQL logging.
- **Optional LLM explanations**: converts deviations into human‑readable rationales when Azure OpenAI config is present.

---

## Architecture

```
Client Request (JSON list of deals)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ infer.py  (KerasAnomalyDetectorPyfunc)                  │
│  • load_context(): artifacts.load_artifacts_from_context│
│  • predict(): preprocess → rules → model → thresholds   │
│  • logs to UC via log_utils                             │
└──────────────────────────────────────────────────────────┘
        │                         ▲
        │ uses                    │ returns
        ▼                         │
┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ artifacts.py   │   │ preprocess.py    │   │ rules.py        │
│ • Load Keras   │   │ • typing, OHE    │   │ • missing group │
│ • Load scalers │   │ • scaling, FV    │   │ • unseen pair   │
│ • Load gaps    │   │ • uniques        │   │ • 12‑month gap  │
└────────────────┘   └──────────────────┘   └─────────────────┘
        ▲                         ▲                     ▲
        │                         │                     │
        ├───────────────┬─────────┴───────────┬─────────┤
        │               │                     │
┌─────────────┐   ┌───────────────┐    ┌───────────────┐
│ safe_pickle │   │ serializers.py │    │ llm_utils.py  │
│ • strict    │   │ • to_utc       │    │ • optional    │
│   unpickler │   │ • to_json      │    │   explain     │
└─────────────┘   └───────────────┘    └───────────────┘
```

**Decision flow:** rules may short‑circuit with a clear reason. Otherwise, the autoencoder scores inputs; deviations are thresholded with three checks (counterparty and feature thresholds) to produce `Anomaly` (`Yes`/`No`) and `Reason` (blank if `No`).

---

## Repository Layout

```
project-root/
  infer.py                # MLflow pyfunc model (entry point)
  utils/artifacts.py            # Artifact loading (model, scalers, year-gap data)
  utils/preprocess.py           # Preprocessing helpers for inference path
  utils/rules.py                # Business rules
  utils/serializers.py          # Timestamp/JSON utilities for logging
  utils/safe_pickle.py          # Restricted unpickler for scaler artifacts
  utils/llm_utils.py            # Optional: LLM-based explanation
  utils/system_prompt.json      # prompt content for llm_utils.py
  utils/env-config.yml          # (not checked in) credentials and runtime env vars
  utils/log_utils.py            # writes logs to Unity Catalog table
```

---

## Quickstart

1. **Install dependencies** (typical conda/pip spec for serving):
   ```yaml
   channels: [conda-forge]
   dependencies:
     - python=3.11
     - pip
     - pip:
       - mlflow==3.0.1
       - tensorflow
       - keras
       - scikit-learn
       - pandas
       - pydantic
       - azure-identity
       - databricks-sql-connector
       - requests
   
   ```

2. **Ensure artifacts exist** (produced by training):
   - `keras_autoencoder_path`: saved Keras model
   - `scalers_path`: pickle containing encoders/scalers and thresholds
   - `year_gap_data_path`: DataFrame with prior 12‑month gaps

3. **Register the model with MLflow** (Databricks registry URI):
   ```python
   import mlflow, infer

   artifacts = {
       "keras_autoencoder_path": "/dbfs/FileStore/models/FISAU/model.keras",
       "scalers_path": "/dbfs/FileStore/models/FISAU/all_scales.pkl",
       "year_gap_data_path": "/dbfs/FileStore/models/FISAU/year_gap_data.pkl",
   }

   mlflow.set_registry_uri("databricks")
   mlflow.pyfunc.log_model(
       python_model=infer.KerasAnomalyDetectorPyfunc(),
       name="fx_model_FISAU",
       conda_env=<conda_env_dict>,
       artifacts=artifacts,
       code_paths=["."],
       registered_model_name="fx_anomaly_prediction_FISAU",
       model_config={"allowed_client": "FISAU"},
   )
   ```

4. **Create/Update a Serving Endpoint** (Databricks UI or SDK), passing environment vars from `env-config.yml` as endpoint environment variables.

5. **Invoke the endpoint**:
   ```bash
   curl -X POST "$DATABRICKS_HOST/serving-endpoints/<endpoint>/invocations" \
     -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     -H "Content-Type: application/json" \
     -d @payload.json
   ```
   Where `payload.json` contains the request as described in [Inference API](#inference-api).

---

## Configuration

**Provided out-of-band via `env-config.yml` (not committed).**

Databricks / Logging:
- `AZURE_TENANT_ID` — AAD tenant
- `WORKSPACE_FQDN` — e.g., `adb-<id>.<region>.azuredatabricks.net`
- `SQL_WAREHOUSE_HTTP_PATH` — SQL warehouse http path
- `AZURE_SP_CLIENT_ID` — service principal app ID
- `AZURE_SP_CLIENT_SECRET` — service principal secret (set as secret)
- `DATABRICKS_SCOPE` — AAD resource scope used to fetch token
- `LOG_TABLE_UC_NAME` — Unity Catalog table to store inference logs

> Set these as environment variables on the Serving Endpoint. Secrets should be injected via secret scopes, not plain text.

---

## Artifacts Produced by Training

The inference service expects these keys when the model is logged:

- **`keras_autoencoder_path`**: Path to the **multi‑input Keras autoencoder** (SavedModel or `.keras`).  
- **`scalers_path`**: Pickle containing for each client (e.g., `"FISAU"`):
  - `ohe`: `sklearn.preprocessing.OneHotEncoder` fitted on training categoricals
  - `mms`: `sklearn.preprocessing.MinMaxScaler` fitted on training numerics
  - `grouped_scalers`: thresholds for `FaceValue` by `(BUnit, Cpty, PrimaryCurr)` → `{"lower": ..., "upper": ...}`
  - `tdays_scalers`: thresholds for `TDays` by `(FX_Instrument_Type,)` → `{"lower": ..., "upper": ...}`
  - `cpty_group`: dict of known `buy`/`sell` currencies by counterparty
- **`year_gap_data_path`**: DataFrame with rows indicating prior 12‑month gaps at the `(BUnit, Cpty, BuyCurr, SellCurr, DealDate)` level.

Deserialization is performed by `safe_pickle.py` to block unsafe classes during unpickling.

---

## Inference API

### Request (JSON)

The endpoint expects a **list** of deals, each validating against the Pydantic schema in `infer.py`:

```json
[
  {
    "Client": "FISAU",
    "FX_Instrument_Type": "FX Forward",
    "BUnit": "0-GLOBAL",
    "Cpty": "CBA",
    "PrimaryCurr": "USD",
    "BuyCurr": "USD",
    "SellCurr": "AUD",
    "BuyAmount": 10000000.0,
    "SellAmount": -10000000.0,
    "SpotRate": 1.0,
    "ForwardPoints": 0.0,
    "DealDate": "2025-09-01",
    "MaturityDate": "2025-09-25",
    "UniqueId": "FX.FWD.Request.100097",
    "ContLeg": 0
  }
]
```

**Constraints**
- `DealDate` and `MaturityDate`: `YYYY-MM-DD`
- `Client` must match `model_config["allowed_client"]`
- `ContLeg` is optional

### Response (per item)

```json
{
  "Client": "FISAU",
  "UniqueId": "FX.FWD.Request.100097",
  "ContLeg": 0,
  "Anomaly": "Yes",
  "Reason": "This currency pair has not been traded in the previous 12 months",
  "status_code": 200
}
```

**Errors**
- `400` with `"error"` on validation/key errors
- `500` with `"error"` on unexpected failures

---

## Deployment — Databricks Model Serving

1. **Register the model** with artifacts and `code_paths` pointing to the folder that contains your modules (`infer.py`, `artifacts.py`, `preprocess.py`, etc.).
2. **Create a Serving Endpoint** targeting the registered model and version. Provide all environment variables from `env-config.yml` as endpoint environment variables (use secret scopes for secrets).
3. **Wait for READY** state; then invoke `/invocations` with a JSON body containing the list of deals under the default PyFunc key (Databricks typically accepts bodies with a top‑level `inputs` field; if your workspace expects that form, wrap the list under `"inputs": [...]`).

Example payload file:
```json
{
  "inputs": [
    { "Client": "FISAU", "...": "..." }
  ]
}
```

---

## Observability

- **Structured Logging**: `log_utils.py` writes one row per request into `LOG_TABLE_UC_NAME` via SQL Warehouse using AAD application credentials. Typical columns include:
  - `log_id`, `log_start_time`, `log_end_time`
  - `model_name`, `model_version`
  - `request_payload`, `response_payload`
  - `api_status`
  - `inference_start_time`, `inference_end_time`
  - `autoencoder_start_time`, `autoencoder_end_time`
- **Serialization**: `serializers.py` provides `to_utc()` and `to_json_safely()` for consistent storage.
- **Optional Explanations**: If AOAI config is present, `llm_utils.py` can turn deviations and context into operator‑friendly reasons.

---

## Security Hardening

- **Restricted Unpickling**: `safe_pickle.py` only allows known‑good classes (e.g., `MinMaxScaler`), and blocks persistent IDs. Re‐export training artifacts using safe types if unpickling fails.
- **Least Privilege**: The service principal used for logging should have minimal UC table permissions and SQL Warehouse access only.
- **Secret Management**: Only set secrets (e.g., `AZURE_SP_CLIENT_SECRET`) via secret scopes; do not commit credentials.
- **Config Validation**: `infer.py` validates the client name at serving time to prevent cross‑tenant leakage of artifacts.

---

## Troubleshooting

- **ImportError (module not found)**: Ensure the folder with these modules is listed in `code_paths` at model log time.
- **pickle.UnpicklingError (Forbidden global)**: Rebuild the scaler pickle using standard sklearn objects only.
- **KeyError loading artifacts**: Verify the three artifact keys (`keras_autoencoder_path`, `scalers_path`, `year_gap_data_path`) exist.
- **400 validation error**: Confirm field names, types, and date formats.
- **Token/permission issues**: Check `AZURE_SP_CLIENT_SECRET`, `DATABRICKS_SCOPE`, SQL Warehouse access, and endpoint environment config.

---

## License

Proprietary — internal use only.
