from __future__ import annotations

import pandas as pd
import mlflow.pyfunc


class BankAnomalyDetectorPyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper around src/inference.py::BankAnomalyDetectorWrapper.

    Expects MLflow artifacts:
      - context.artifacts["model_dir"] -> directory containing:
          - the .keras model
          - meta.json, combo_map.json, combo_stats.json, combo_threshold.json
          - optional account_map.json / bu_map.json / code_map.json
    """

    def load_context(self, context):
        # Packaged from code_paths=["src"]
        from inference import BankAnomalyDetectorWrapper

        model_dir = context.artifacts["model_dir"]
        self.detector = BankAnomalyDetectorWrapper(model_dir=model_dir)

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(model_input)}")
        return self.detector.score_bank_statement_df(model_input)
