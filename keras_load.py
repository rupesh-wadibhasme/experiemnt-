class KerasAnomalyDetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the Keras Autoencoder model and all preprocessors/scalers.
        Assumes these are saved as artifacts in the MLflow run.
        """
        actual_keras_file_path = context.artifacts["keras_autoencoder_path"]
        self.model = tf.keras.models.load_model(actual_keras_file_path)
        self.allowed_client_for_this_endpoint = context.model_config.get("allowed_client", "default")

        # Load the dictionary of scalers and encoders
        with open(context.artifacts["scalers_path"], 'rb') as f:
            self.load_scalers = pickle.load(f)[self.allowed_client_for_this_endpoint]

        


        
        # Pre-extract unique values from scalers for faster lookup
        self.unique_BU, self.unique_cpty, self.unique_primarycurr = self._get_uniques(self.load_scalers['grouped_scalers'])

        # Initialize the Pydantic validator
        # self.ValidateParams = ValidateParams
        # logger.info("Model and scalers loaded successfully.")
