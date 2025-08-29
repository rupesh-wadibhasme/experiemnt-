# Databricks notebook source
# MAGIC %md
# MAGIC ## Custom Pyfunc Model Serving Example
# MAGIC
# MAGIC - Exploring logging options
# MAGIC

# COMMAND ----------

#!pip install requests==2.32.3 --quiet
%pip install azure-identity==1.23.0 --quiet
%pip install databricks-sql-connector==4.0.3 --quiet
%pip install mlflow-skinny[databricks]==3.3.1 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.deployments import get_deploy_client

import pandas as pd
from typing import Dict, Any


# COMMAND ----------

print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Table to which API invocations will be logged

# COMMAND ----------

from pyspark.sql import SparkSession

TABLE_NAME = "aimluat.treasury_anomaly_fx_detection.inference_logs_v3"

#
# can use this idiom to create tables.
# can use default current_timestamp in log entries to save time/energy in the code that needs to log various entries.

sql_create_stmt = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    log_ts TIMESTAMP DEFAULT current_timestamp(),
    log_entry STRING
)
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported');
"""

spark_session = SparkSession.builder.getOrCreate()
spark_session.sql(sql_create_stmt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Implement the Custom "Model" in Databricks
# MAGIC
# MAGIC - The predict method will be invoked each time the correspoding Model Serving operation is invoked.
# MAGIC - The predict method will use Databricks python SQL connector to read Delta tables.

# COMMAND ----------

from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Tuple

from azure.identity import ClientSecretCredential
from databricks.sql import connect

#
# all the these env vars need to be specified during model deployment ... see below.
# will it work if I put these into a .env file?

#
# prod.cloud tenant
os.environ["AZURE_TENANT_ID"] = "ca9"

#
# this is needed for connection to SQL Warehouse and invoking model serving endpoint
os.environ["WORKSPACE_FQDN"] = "adb-85209.azuredatabricks.net"

#
# this corresponds to SQL Warehouse = aiml-test-sql-warehouse in dbrick-dev
os.environ["SQL_WAREHOUSE_HTTP_PATH"] = "/sql/1.0/warehouses/c38a08c"

#
# this is the client id for the Azure Entra Service Principal that is used to authenticate/authorize
# queries against Delta tables via SQL Warehouse.  This SP will also be used to invoke the Model Serving
# endpoint.
os.environ["AZURE_SP_CLIENT_ID"] = "95b4e7-2b6f2bcd7d39"
os.environ["AZURE_SP_CLIENT_SECRET"] = dbutils.secrets.get(scope='CDPKDEV2', key='sp-aims-anofxj-uat')
os.environ["LOG_TABLE_UC_NAME"] = TABLE_NAME


class ExamplePyfuncModel(PythonModel):

    def __init__(self):
        self.host_name = os.environ["WORKSPACE_FQDN"]
        self.warehouse_http_path = os.environ["SQL_WAREHOUSE_HTTP_PATH"]
        self.log_table_name = os.environ["LOG_TABLE_UC_NAME"]
        #
        # this can be the same across all Workspaces (i.e., dev, uat, prod)
        self.endpoint_name = "example-endpoint"
        self.model_name = "example-model"
        #
        # declare dependencies so they can be used when logging the model
        self.requirements = ["databricks-sql-connector==4.0.3", "azure-identity==1.23.0"]


    class InputArgs():

        def __init__(self, model_input: pd.DataFrame):
            self.input_message = None
            try:
                self.input_message = model_input.iloc[0,0]
            except Exception as e:
                raise RuntimeError("unable to understand request")

        def is_valid(self) -> Tuple[bool, str]:
            len_input_message = len(self.input_message)
            if not (len_input_message > 0 and len_input_message <= 64 and self.input_message.isalnum()):
                return False, "Invalid input message"

            return True, None


    def entra_access_token(self) -> Tuple[bool, str]:
        try:
            credential = ClientSecretCredential(
                client_id=os.environ["AZURE_SP_CLIENT_ID"], 
                client_secret=os.environ["AZURE_SP_CLIENT_SECRET"],
                tenant_id=os.environ["AZURE_TENANT_ID"]
            )
            #
            # this should be the same across all environments
            databricks_scope = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
            token_response = credential.get_token(databricks_scope)
            assert token_response
            return (True, token_response.token)
        except Exception as e:
            return (False, e.message)


    def create_log_record(self, cursor, message):
        #
        # https://learn.microsoft.com/en-us/azure/databricks/dev-tools/python-sql-connector#query-data
        cursor.execute(
            f"insert into {self.log_table_name} (log_entry) values ('{message}')"
        )


    def execute(self, cursor, model_input: pd.DataFrame) -> Tuple[bool, str, dict]:
        try:
            self.create_log_record(cursor, message=f"test_invoking...")
            args = self.InputArgs(model_input=model_input)
            ok, err_message = args.is_valid()
            if not ok:
                self.create_log_record(cursor, message=f"invalid arguments {err_message}")
                return False, err_message, None

            #
            # log the request as it was made
            json_args = model_input.to_json(orient='records')
            self.create_log_record(cursor, message=f"invoked with {json_args}")
            
            return True, None, { "result": f"you said {args.input_message}"}
        except Exception as e:
            #
            # something bad happened ...log it, and re-raise
            self.create_log_record(cursor, message=f"exception {e.message}")
            raise


    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> list[dict[str, Any]]:
        #
        # if this fails, then we can't log anything ... since we can't get a connection to 
        # the SQL warehouse
        ok, access_token = self.entra_access_token()
        if not ok:
            #
            # this causes a 400 to be returned ... what we really need in this case is a 500 ... this does not work, i.e., an 
            # http 400 is returned
            raise mlflow.exceptions.MlflowException(f"cannot get access token", error_code=mlflow.exceptions.get_error_code(500))

        try:
            with connect(
                server_hostname  = self.host_name,
                http_path        = self.warehouse_http_path,
                access_token     = access_token
            ) as connection:
                with connection.cursor() as cursor:
                    ok, err_message, response = self.execute(cursor, model_input=model_input)
                    if not ok:
                        raise mlflow.exceptions.MlflowException(err_message, error_code=mlflow.exceptions.get_error_code(400))
                    return response
        except Exception as e:
            #
            # this will likely return a 500 error
            raise mlflow.exceptions.MlflowException(f"unexpected error", error_code=mlflow.exceptions.get_error_code(500))


# COMMAND ----------


#
# this works!
model_object = ExamplePyfuncModel()
input_example = pd.DataFrame({"input_message": ["hello"] })
response = model_object.predict(context=None, model_input=input_example)
assert response

# COMMAND ----------

response

# COMMAND ----------

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel, log_model

#
# don't log to UC ... Serverless Compute (i.e., Model Serving endpoint) cannot talk to Storage Account associated with UC.
mlflow.set_registry_uri("databricks")

def log_serving_model(serving_model:PythonModel, input_example: pd.DataFrame):
    """ 
    Log a Serving Model to MLflow
    """
    with mlflow.start_run():
        serving_model: ModelInfo = log_model(
            name = serving_model.model_name,
            python_model=serving_model,
            input_example=input_example,
            pip_requirements=serving_model.requirements,
            registered_model_name = serving_model.model_name
        )
        return serving_model


# COMMAND ----------

model_info: ModelInfo = log_serving_model(model_object, input_example)

# COMMAND ----------

import os
from mlflow.deployments import get_deploy_client

def deploy_serving_model(serving_model: PythonModel, version: str):
    client = get_deploy_client("databricks")
    endpoint = client.create_endpoint(
        name=serving_model.endpoint_name,
        config={
            "name": serving_model.endpoint_name,
            "served_entities": [
                {
                    "name": serving_model.endpoint_name,
                    "entity_name":  serving_model.model_name,
                    "entity_version": version,
                    "scale_to_zero_enabled": True,
                    "workload_size": "Small",
                    "workload_type": "CPU",
                    "scale_to_zero_enabled": True,
                    "environment_vars": {
                        "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
                        "AZURE_SP_CLIENT_ID": os.getenv("AZURE_SP_CLIENT_ID"),
                        "AZURE_SP_CLIENT_SECRET": os.getenv("AZURE_SP_CLIENT_SECRET"),
                        "WORKSPACE_FQDN": os.getenv("WORKSPACE_FQDN"),
                        "SQL_WAREHOUSE_HTTP_PATH": os.getenv("SQL_WAREHOUSE_HTTP_PATH"),
                        "LOG_TABLE_UC_NAME": os.getenv("LOG_TABLE_UC_NAME")
                    }
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_entity_name": serving_model.endpoint_name,
                        "served_model_name": serving_model.model_name,
                        "traffic_percentage": 100
                    }
                ]
            }
        }
    )

# COMMAND ----------

deploy_serving_model(model_object, version="2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Invoke Model Serving API
# MAGIC
# MAGIC - Uses Azure Service Principal to authenticate/authorize invocations via Entra.
# MAGIC - There are multiple techniques to authenticate/authorize invocations, but this best simulates how CM Integrity will invoke the endpoint.
# MAGIC - See https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/oauth-m2m

# COMMAND ----------

from azure.core.credentials import TokenCredential
import requests

def invoke_serving_endpoint(token_povider: TokenCredential, url: str, request_body: dict):
    databricks_scope = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
    token = token_povider.get_token(databricks_scope)
    response = requests.post(
        url = url,
        headers = { 'Authorization': f"Bearer {token.token}"},
        json=request_body,
    )
    return response

# COMMAND ----------

azure_sp_client_secret = dbutils.secrets.get(scope="CDPKEYVAULTDEV2", key="sp-aiml-ntgrty-uat")
assert azure_sp_client_secret

# COMMAND ----------

import os
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
    #
    # This is "Application ID" in Entra Enterprise Application for "sp-aiml-dbricks-intgrty-uat" Service Principal
    client_id="c3c9c6cc-8333-813419e22150", 
    client_secret=azure_sp_client_secret,
    tenant_id="ca97f7f3-edf2b862856c"
)

endpoint = os.getenv("WORKSPACE_FQDN")
assert endpoint

response = invoke_serving_endpoint(
    credential,
    f"https://{endpoint}/serving-endpoints/{model_object.endpoint_name}/invocations", 
    { "dataframe_records": [{"input_message": "hello" }]}
)
assert response.status_code == 200, f"Error: {response.status_code} - {response.text}"

# COMMAND ----------

response.json()


    

