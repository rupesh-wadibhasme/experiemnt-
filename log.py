# --- BEGIN: drop-in event logger (4-column table) ---
import os, json
import pandas as pd
from databricks.sql import connect
from azure.identity import ClientSecretCredential

# UC table with these exact columns: log_ts (DEFAULT), model_name, request_payload, response_payload, status
LOG_TABLE_UC_NAME = os.getenv("LOG_TABLE_UC_NAME", "aimluat.treasury_anomaly_fx_detection.inference_logs_v3")

# Required env (same as your experimental script)
WORKSPACE_FQDN           = os.getenv("WORKSPACE_FQDN")            # e.g. "adb-xxxx.azuredatabricks.net"
SQL_WAREHOUSE_HTTP_PATH  = os.getenv("SQL_WAREHOUSE_HTTP_PATH")   # e.g. "/sql/1.0/warehouses/xxxxx"
AZURE_TENANT_ID          = os.getenv("AZURE_TENANT_ID")
AZURE_SP_CLIENT_ID       = os.getenv("AZURE_SP_CLIENT_ID")
AZURE_SP_CLIENT_SECRET   = os.getenv("AZURE_SP_CLIENT_SECRET")
_SCOPE                   = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"  # Databricks resource

def _get_access_token() -> str:
    cred = ClientSecretCredential(
        client_id=AZURE_SP_CLIENT_ID,
        client_secret=AZURE_SP_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID
    )
    return cred.get_token(_SCOPE).token

def _to_json(payload) -> str:
    """Safely JSON-serialize dict/str/DataFrame/ndarray/etc."""
    try:
        if isinstance(payload, pd.DataFrame):
            payload = json.loads(payload.to_json(orient="records"))
        return json.dumps(payload, default=str)
    except Exception:
        try:
            return json.dumps(str(payload))
        except Exception:
            return '"<unserializable>"'

def log_event(model_name: str, request_payload=None, response_payload=None, status: str = "INFO"):
    """
    Insert one row into:
      (log_ts DEFAULT current_timestamp(), model_name, request_payload, response_payload, status)

    Args:
      model_name: e.g. f"{REGISTERED_CLIENT_MODEL_NAME_BASE}_{client}"
      request_payload / response_payload: dict | str | pandas.DataFrame | numpy types (auto-JSON)
      status: 'SUCCESS' | 'FAILURE' | 'INFO' | etc.

    Non-fatal: prints a warning if logging fails.
    """
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN,
                     http_path=SQL_WAREHOUSE_HTTP_PATH,
                     access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {LOG_TABLE_UC_NAME}
                        (model_name, request_payload, response_payload, status)
                        VALUES (?, ?, ?, ?)""",
                    (model_name, _to_json(request_payload), _to_json(response_payload), status)
                )
    except Exception as e:
        print(f"[log_event] WARN: failed to write log: {e}")
# --- END: drop-in event logger ---


except Exception as e:
    log_event(
        model_name=f"{REGISTERED_CLIENT_MODEL_NAME_BASE}_{client}" if 'client' in locals() else "unknown",
        request_payload={"stage":"TRAINING"},
        response_payload={"error": str(e)},
        status="FAILURE"
    )
    raise


log_event(
    model_name=client_registered_model_full_name,
    request_payload={"stage":"REGISTRATION","client":client},
    response_payload={"run_id": run_id},
    status="SUCCESS"
)


