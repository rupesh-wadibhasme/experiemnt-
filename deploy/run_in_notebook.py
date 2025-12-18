import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature

from src.mlflow_pyfunc_bank_anomaly import BankAnomalyDetectorPyFunc
from src.inference import BankAnomalyDetectorWrapper


def log_bank_anomaly_model(
    *,
    artifacts_dir: str = "artefacts",
    df_sample: pd.DataFrame,
    run_name: str = "bank_anomaly",
    registered_model_name: str | None = "bank_anomaly_detector",
):
    """
    Logs your inference pipeline as an MLflow pyfunc model.

    artifacts_dir: folder containing trained model (.keras) + required jsons
    df_sample: sample raw input dataframe for signature inference
    """

    # 1) Build signature using your actual inference pipeline
    detector = BankAnomalyDetectorWrapper(model_dir=artifacts_dir)
    input_sample = df_sample.head(5)
    output_sample = detector.score_bank_statement_df(input_sample)
    signature = infer_signature(input_sample, output_sample)

    # 2) Log pyfunc model + pack artefacts folder + include src/
    with mlflow.start_run(run_name=run_name):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BankAnomalyDetectorPyFunc(),
            artifacts={"model_dir": artifacts_dir},   # logs entire directory
            code_paths=["src"],                       # include src support scripts
            signature=signature,
            registered_model_name=registered_model_name,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                "tensorflow>=2.13.0",
                "pandas>=1.5.0",
                "numpy>=1.23.0",
                "scikit-learn>=1.2.0",
                "pyyaml>=6.0",
                "cloudpickle>=2.2.0",
            ],
        )
        print("Logged MLflow pyfunc model at artifact_path='model'")


# ----------------

# df is your raw bank-statement dataframe (same schema you pass to score_bank_statement_df)
log_bank_anomaly_model(artifacts_dir="artefacts", df_sample=df)

# --------
run_id = mlflow.active_run().info.run_id  # if still inside the run context
m = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
pred = m.predict(df.head(10))
print(pred.head())
