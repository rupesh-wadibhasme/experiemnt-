# Databricks runtime ≥ 13.x
%pip install --quiet openpyxl mlflow==2.12.1 pyyaml tensorflow==2.16.1

import os, json, yaml, pickle, tempfile, datetime, matplotlib.pyplot as plt
import numpy as np, pandas as pd, mlflow, tensorflow as tf
from pyspark.sql import SparkSession
spark = SparkSession.getActiveSession()

dbutils.widgets.text("client",   "ixom")
dbutils.widgets.text("raw_vol",  "/Volumes/ai/raw")        # Excel upload path
dbutils.widgets.text("bronze_vol","/Volumes/ai/bronze")    # Delta output
client     = dbutils.widgets.get("client").lower()
raw_path   = f"{dbutils.widgets.get('raw_vol')}/{client}"
bronze_tbl = f"{dbutils.widgets.get('bronze_vol')}/{client}/training_delta"
cfg_path   = f"/Workspace/Repos/<your-repo>/configs/{client}.yaml"

cfg = yaml.safe_load(open(cfg_path))


3. Cell 3 – paste feature-engineering helpers
Simply drop every function from your “data prep script” here (or keep them in a helper notebook and # MAGIC %run ./shared/feat_eng).



with mlflow.start_run(run_name=f"prep_{client}"):

    # 1. Excel → Delta (skip if Delta already exists)
    if not spark._jsparkSession.catalog().tableExists(bronze_tbl):
        df = (spark.read.format("com.crealytics.spark.excel")
                  .option("header","true").option("inferSchema","true")
                  .load(f"{raw_path}/*.xlsx"))
        (df.write.mode("overwrite").format("delta").save(bronze_tbl))

    pdf = (spark.read.format("delta").load(bronze_tbl)).toPandas()

    # 2. All your original feature-eng code …
    pdf = drop_unwanted_data(pdf, drop_cols_master)
    pdf['Instrument'] = pdf['Instrument'].str.upper().str.replace(r'\s+','',regex=True)
    pdf, cat_cols, num_cols = add_derived_features(pdf)
    #  …etc…

    features, ohe, mms = build_feature_frame(pdf, cat_cols, num_cols,
                                             output_path="/tmp", CLIENT_NAME=client.upper())

    # 3. Log artefacts for repeatable inference
    mlflow.log_artifacts("/tmp", artifact_path="preproc")

    # 4. Build model-ready blocks
    blocks = [features[[c for c in features if c.startswith(col)]]
              for col in cat_cols] + [features[num_cols]]


5. Cell 5 – paste model-builder helpers
Drop everything from “training script” up through build_embed_autoencoder, prepare_blocks, etc.

# ---

numeric_block_idx = len(cat_cols)
cat_arrays, num_array, cardinals, embed_dims = prepare_blocks(blocks, numeric_block_idx)

inputs  = cat_arrays + [num_array]
targets = inputs

with mlflow.start_run(run_name=f"train_{client}", nested=True):
    mlflow.log_params(cfg)

    model = build_embed_autoencoder(
        cardinals, num_dim=num_array.shape[1], embed_dims=embed_dims,
        hid=cfg.get("hid",64), bottleneck=cfg.get("bottleneck",32),
        dropout=cfg.get("dropout",0.15))

    early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(inputs, targets,
                        validation_split=0.3,
                        epochs=cfg.get("num_epochs",100),
                        batch_size=cfg.get("batch_size",64),
                        callbacks=[early],
                        verbose=2)

    mlflow.keras.log_model(
        model, artifact_path="model",
        registered_model_name=f"ae_{client}"
    )

    for k,v in history.history.items():
        mlflow.log_metric(k, v[-1])

# -----

numeric_block_idx = len(cat_cols)
cat_arrays, num_array, cardinals, embed_dims = prepare_blocks(blocks, numeric_block_idx)

inputs  = cat_arrays + [num_array]
targets = inputs

with mlflow.start_run(run_name=f"train_{client}", nested=True):
    mlflow.log_params(cfg)

    model = build_embed_autoencoder(
        cardinals, num_dim=num_array.shape[1], embed_dims=embed_dims,
        hid=cfg.get("hid",64), bottleneck=cfg.get("bottleneck",32),
        dropout=cfg.get("dropout",0.15))

    early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(inputs, targets,
                        validation_split=0.3,
                        epochs=cfg.get("num_epochs",100),
                        batch_size=cfg.get("batch_size",64),
                        callbacks=[early],
                        verbose=2)

    mlflow.keras.log_model(
        model, artifact_path="model",
        registered_model_name=f"ae_{client}"
    )

    for k,v in history.history.items():
        mlflow.log_metric(k, v[-1])

# ----
# 7. Cell 7 – smoke-test Production model

prod_model = mlflow.pyfunc.load_model(f"models:/ae_{client}/Latest")
sample     = pd.DataFrame(num_array[:5])        # trivial example
prod_model.predict(sample)

