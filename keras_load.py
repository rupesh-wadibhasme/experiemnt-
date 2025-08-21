# --- Add these imports at the top of your file ---
import pickle
import builtins

# --- Safe Unpickler (drop-in) ---
class _RestrictedUnpickler(pickle.Unpickler):
    """Only allow harmless built-ins; block everything else (prevents RCE)."""
    _ALLOWED = {"dict", "list", "tuple", "set", "str", "int", "float", "bool", "bytes"}

    def find_class(self, module, name):
        if module == "builtins" and name in self._ALLOWED:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"Forbidden global during unpickling: {module}.{name}")

    # Disallow external persistent references
    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent IDs are forbidden")

def _safe_pickle_load(file_obj):
    return _RestrictedUnpickler(file_obj).load()


# --- Replace your existing load_context with this version (only change is safe unpickle) ---
class KerasAnomalyDetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the Keras Autoencoder model and all preprocessors/scalers safely.
        """
        actual_keras_file_path = context.artifacts["keras_autoencoder_path"]
        self.model = tf.keras.models.load_model(actual_keras_file_path)
        self.allowed_client_for_this_endpoint = context.model_config.get("allowed_client", "default")

        # SAFE: restricted unpickle of dict/list-of-dicts only
        with open(context.artifacts["scalers_path"], "rb") as f:
            scalers_all = _safe_pickle_load(f)

        self.load_scalers = scalers_all[self.allowed_client_for_this_endpoint]

        # Pre-extract unique values from scalers for faster lookup
        self.unique_BU, self.unique_cpty, self.unique_primarycurr = self._get_uniques(
            self.load_scalers['grouped_scalers']
        )

