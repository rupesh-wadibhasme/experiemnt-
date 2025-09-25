import json
import pandas as pd
import yaml
from databricks.sql import connect
from azure.identity import ClientSecretCredential
# from databricks.sdk.runtime import *
import os
import mlflow

with open("utils/env-config.yml", "r") as f:
    config = yaml.safe_load(f)["env"]

# os.environ["AZURE_SP_CLIENT_SECRET"] = dbutils.secrets.get(scope=config["DATABRICKS_SECRET_SCOPE"], key=config["DATABRICKS_SECRET_KEY"])

TABLE_BATCH_LEVEL = config["TABLE_BATCH_LEVEL"]
TABLE_TASK_LEVEL = config["TABLE_TASK_LEVEL"]
DATABRICKS_SCOPE = config["DATABRICKS_SCOPE"]
AZURE_TENANT_ID = config["AZURE_TENANT_ID"]
WORKSPACE_FQDN = config["WORKSPACE_FQDN"]
SQL_WAREHOUSE_HTTP_PATH = config["SQL_WAREHOUSE_HTTP_PATH"]
AZURE_SP_CLIENT_ID = config["AZURE_SP_CLIENT_ID"]
AZURE_SP_CLIENT_SECRET = os.getenv("AZURE_SP_CLIENT_SECRET")
LOG_TABLE_BATCH_LEVEL = config["TABLE_BATCH_LEVEL"]
LOG_TABLE_TASK_LEVEL = config["TABLE_TASK_LEVEL"]
DATABRICKS_HOST = config['DATABRICKS_HOST']
STATUS_TABLE_NAME = config["STATUS_TABLE_NAME"]
LOG_TABLE_UC_NAME = config["LOG_TABLE_UC_NAME"]

DATABRICKS_SECRET_SCOPE = config["DATABRICKS_SECRET_SCOPE"]
DATABRICKS_SECRET_KEY = config["DATABRICKS_SECRET_KEY"]
BASE_VOLUME_PATH = config["BASE_VOLUME_PATH"]
MODEL_NAME_BASE = config["MODEL_NAME_BASE"]

os.environ["AOAI_ENDPOINT"] = config["AOAI_ENDPOINT"]
os.environ['OPENAI_SCOPE'] = config["OPENAI_SCOPE"]

var_azure_secret="{{secrets/"+f"{DATABRICKS_SECRET_SCOPE}"+'/'+f"{DATABRICKS_SECRET_KEY}"+"}}"


def _get_access_token() -> str:
    
    # AZURE_SP_CLIENT_SECRET = os.getenv("AZURE_SP_CLIENT_SECRET")
    # if not client_secret:
    #     raise RuntimeError("AZURE_SP_CLIENT_SECRET is not set. Please configure it as an environment variable.")
    
    cred = ClientSecretCredential(
        client_id=AZURE_SP_CLIENT_ID,
        client_secret=AZURE_SP_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID
    )
    return cred.get_token(DATABRICKS_SCOPE).token

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
        
def log_event(
    log_id: int,
    log_start_time=None,
    log_end_time=None,
    model_name: str = None,
    model_version: str = None,
    request_payload=None,
    response_payload=None,
    api_status: str = None,
    inference_start_time=None,
    inference_end_time=None,
    autoencoder_start_time=None,
    autoencoder_end_time=None,
):
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN, http_path=SQL_WAREHOUSE_HTTP_PATH, access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {LOG_TABLE_UC_NAME}
                        (log_id, log_start_time, log_end_time, model_name, model_version,
                         request_payload, response_payload, api_status,
                         inference_start_time, inference_end_time, autoencoder_start_time, autoencoder_end_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_id, _ts(log_start_time), _ts(log_end_time), model_name, model_version,
                        _to_json(request_payload), _to_json(response_payload), api_status,
                        _ts(inference_start_time), _ts(inference_end_time),
                        _ts(autoencoder_start_time), _ts(autoencoder_end_time),
                    ),
                )
    except Exception as e:
        print(f"[log_event] WARN: failed to write log: {e}")

def log_job(job_id: int, tenant_id: str,job_start_time,job_end_time,
              job_status:str="PENDING"
              ):
    """
    Args:
      job_id: int 
      tenant_id: string eg. 'RCC','IXOM'
      job_start_time: timestamp
      job_end_time: timestamp
      job_status: string e.g. 'PENDING','COMPLETED'
    """
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN,
                     http_path=SQL_WAREHOUSE_HTTP_PATH,
                     access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {LOG_TABLE_BATCH_LEVEL}
                        (job_id, tenant_id,job_start_time,job_end_time,job_status)
                        VALUES (?, ?, ?, ?, ?)""",
                    (job_id, tenant_id,job_start_time,job_end_time,job_status)
                 )

    except Exception as e:
        print(f"[log_job] WARN: failed to write log: {e}")


def log_job_detail(job_id: int, tenant_id: str,
                   step_name:str,step_start_time, step_end_time, 
                   step_status:str="PENDING",
                   log_msg:str=""
                    ):
    """
    Args:
      job_id: int e.g. 'RCC','IXOM'
      tenant_id: string
      step_name: string
      step_start_time: timestamp
      step_end_time: timestamp
      step_status: string e.g. 'PENDING','COMPLETED'
      log_msg: string
    """
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN,
                     http_path=SQL_WAREHOUSE_HTTP_PATH,
                     access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {LOG_TABLE_TASK_LEVEL}
                        (job_id, tenant_id,step_name,step_start_time,step_end_time,step_status,log_msg)
                        VALUES (?, ?, ?, ?, ?,?,?)""",
                    (job_id, tenant_id,step_name,step_start_time,step_end_time,step_status,log_msg)
                 )

    except Exception as e:
        print(f"[log_job_detail] WARN: failed to write log: {e}")


def get_job_status_from_steps(job_id: int, tenant_id: str) -> str:
    """
    Checks the step statuses for a given job_id and tenant_id in log_job_detail table.
    Returns 'FAILED' if any step failed, 'COMPLETED' if all steps completed, else 'PENDING'.
    """
    try:
        token = _get_access_token()
        with connect(server_hostname=WORKSPACE_FQDN,
                     http_path=SQL_WAREHOUSE_HTTP_PATH,
                     access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT step_status FROM {LOG_TABLE_TASK_LEVEL}
                        WHERE job_id = ? AND tenant_id = ?""",
                    (job_id, tenant_id)
                )
                statuses = [row[0] for row in cur.fetchall()]

                if not statuses:
                    print(f"[get_job_status_from_steps] No steps found for job_id={job_id}, tenant_id={tenant_id}")
                    return "FAILED"

                if any(status == "FAILED" for status in statuses):
                    return "FAILED"
                elif all(status == "COMPLETED" for status in statuses):
                    return "COMPLETED"
                else:
                    return "FAILED"

    except Exception as e:
        print(f"[get_job_status_from_steps] WARN: failed to fetch step statuses: {e}")
        return "FAILED"  # Default to FAILED in case of error
