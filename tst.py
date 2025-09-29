import os
import sys
import io
import json
import pickle
import types
import unittest
import tempfile
import importlib
import contextlib
import numpy as np
import pandas as pd

# --- Ensure project root on path ---
# If this file is under project_root/test/, parent-of-parent is the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Prepare stub modules for odd import paths used in artifacts.py ---
# artifacts.py may do: from fx_serving.serving_security.safe_pickle import safe_load
# We'll create a stub package hierarchy and redirect safe_load to our local utils.safe_pickle.safe_load
stub_pkg = types.ModuleType("fx_serving")
stub_serving_security = types.ModuleType("fx_serving.serving_security")
stub_safe_pickle = types.ModuleType("fx_serving.serving_security.safe_pickle")

# Bind local safe_load implementation if available; else provide a simple passthrough
local_sp = None
for candidate in ("utils.safe_pickle", "safe_pickle"):
    try:
        local_sp = importlib.import_module(candidate)
        break
    except Exception:
        pass

def _safe_load(file_obj):
    if local_sp and hasattr(local_sp, "safe_load"):
        return local_sp.safe_load(file_obj)
    return pickle.load(file_obj)

setattr(stub_safe_pickle, "safe_load", _safe_load)
sys.modules["fx_serving"] = stub_pkg
sys.modules["fx_serving.serving_security"] = stub_serving_security
sys.modules["fx_serving.serving_security.safe_pickle"] = stub_safe_pickle

# --- Import target modules from utils package ---
serializers = importlib.import_module("utils.serializers")
safe_pickle_mod = importlib.import_module("utils.safe_pickle")
rules = importlib.import_module("utils.rules")
preprocess = importlib.import_module("utils.preprocess")

class TestSerializers(unittest.TestCase):
    def test_to_utc_various_inputs(self):
        from datetime import datetime, timezone, timedelta
        # epoch int
        dt = serializers.to_utc(0)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(int(dt.timestamp()), 0)

        # float epoch
        dt2 = serializers.to_utc(1.5)
        self.assertEqual(dt2.tzinfo, timezone.utc)
        self.assertEqual(int(dt2.timestamp()), 1)

        # ISO string with Z
        dt3 = serializers.to_utc("1970-01-01T00:00:01Z")
        self.assertEqual(int(dt3.timestamp()), 1)

        # naive datetime becomes UTC
        naive = datetime(1970,1,1,0,0,2)
        dt4 = serializers.to_utc(naive)
        self.assertEqual(dt4.tzinfo, timezone.utc)
        self.assertEqual(int(dt4.timestamp()), 2)

        # tz-aware stays as-is
        aware = datetime(1970,1,1,0,0,3, tzinfo=timezone.utc)
        dt5 = serializers.to_utc(aware)
        self.assertIs(aware, dt5)

        # empty & unknown -> None
        self.assertIsNone(serializers.to_utc(""))
        self.assertIsNone(serializers.to_utc(object()))

    def test_to_json_safely(self):
        # Dict
        out = serializers.to_json_safely({"a": 1})
        self.assertEqual(json.loads(out), {"a": 1})

        # DataFrame
        df = pd.DataFrame([{"x": 1}, {"x": 2}])
        out_df = serializers.to_json_safely(df)
        self.assertEqual(json.loads(out_df), [{"x": 1}, {"x": 2}])

        # Unserializable -> string fallback
        class X: pass
        x = X()
        out_x = serializers.to_json_safely(x)
        self.assertIsInstance(out_x, str)
        # Must be valid JSON string
        json.loads(out_x)

class Evil(object):
    def __init__(self):
        self.x = 42

class TestSafePickle(unittest.TestCase):
    def test_safe_load_allows_minmaxscaler(self):
        # Create a legitimate MinMaxScaler pickle and ensure safe_load can read it
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        with tempfile.TemporaryFile() as f:
            pickle.dump(scaler, f)
            f.seek(0)
            obj = safe_pickle_mod.safe_load(f)
        self.assertIsInstance(obj, MinMaxScaler)

    def test_safe_load_blocks_custom_class(self):
        # Define a custom class, pickle an instance, and ensure safe_load rejects it
        with tempfile.TemporaryFile() as f:
            pickle.dump(Evil(), f)
            f.seek(0)
            with self.assertRaises(Exception):
                safe_pickle_mod.safe_load(f)

class TestRules(unittest.TestCase):
    def test_compare_rows_true_and_false(self):
        df1 = pd.DataFrame([
            {"BUnit":"A","Cpty":"C1","BuyCurr":"USD","SellCurr":"INR","DealDate":"2024-01-01"},
            {"BUnit":"B","Cpty":"C2","BuyCurr":"EUR","SellCurr":"USD","DealDate":"2024-02-01"},
        ])
        df2_match = pd.DataFrame([{"BUnit":"B","Cpty":"C2","BuyCurr":"EUR","SellCurr":"USD","DealDate":"2024-02-01"}])
        df2_nomatch = pd.DataFrame([{"BUnit":"X","Cpty":"C9","BuyCurr":"JPY","SellCurr":"GBP","DealDate":"2024-03-01"}])

        self.assertTrue(rules.compare_rows(df1, df2_match))
        self.assertFalse(rules.compare_rows(df1, df2_nomatch))

    def test_check_missing_group_various(self):
        data = pd.DataFrame([{"BUnit":"A","Cpty":"C1","PrimaryCurr":"USD"}])

        # Case 1: No grouped_scalers present -> missing
        scalers = {}
        code, _ = rules.check_missing_group(data, scalers, set(), set(), set())
        self.assertEqual(code, 1)

        # Case 2: grouped_scalers exists but group not seen
        scalers = {"grouped_scalers": {("B","C2","EUR"): {}}}
        code, _ = rules.check_missing_group(data, scalers, {"B"}, {"C2"}, {"EUR"})
        self.assertEqual(code, 1)

        # Case 3: only BU missing
        scalers = {"grouped_scalers": {("A","C1","USD"): {}}}
        code, _ = rules.check_missing_group(
            data, scalers, set(), {"C1"}, {"USD"}
        )
        self.assertEqual(code, 1)

        # Case 4: none missing
        code, _ = rules.check_missing_group(
            data, scalers, {"A"}, {"C1"}, {"USD"}
        )
        self.assertEqual(code, 0)

    def test_check_currency(self):
        # Case: cpty not present at all
        data = pd.DataFrame([{"Cpty":"C1","BuyCurr":"USD","SellCurr":"INR"}])
        scalers = {"cpty_group": {}}
        code, _ = rules.check_currency(data, scalers)
        self.assertEqual(code, 1)

        # Case: present but missing both buy/sell
        scalers = {"cpty_group": {"C1": {"buy": {"EUR": {}}, "sell": {"GBP": {}}}}}
        code, _ = rules.check_currency(data, scalers)
        self.assertEqual(code, 1)

        # Case: present, missing buy only
        scalers = {"cpty_group": {"C1": {"buy": {"EUR": {}}, "sell": {"INR": {}}}}}
        code, _ = rules.check_currency(data, scalers)
        self.assertEqual(code, 1)

        # Case: present, no missing
        scalers = {"cpty_group": {"C1": {"buy": {"USD": {}}, "sell": {"INR": {}}}}}
        code, _ = rules.check_currency(data, scalers)
        self.assertEqual(code, 0)

class TestPreprocess(unittest.TestCase):
    def test_get_column_types(self):
        df = pd.DataFrame({
            "cat": ["x","y","z"],
            "num1": [1.0, 2.0, 3.0],
            "num2": [1, 2, 3],
        })
        cats, nums = preprocess._get_column_types(df)
        self.assertIn("cat", cats)
        self.assertIn("num1", nums)
        self.assertIn("num2", nums)

    def test_one_hot_with_sklearn_encoder(self):
        df = pd.DataFrame({"color": ["red","blue","red","green"]})
        from sklearn.preprocessing import OneHotEncoder
        # use 'sparse' for older sklearn versions
        enc = OneHotEncoder(sparse=False).fit(df)
        enc_df = preprocess._one_hot(df, enc)
        # Should have 3 columns (one per unique)
        self.assertEqual(enc_df.shape[1], 3)
        # Should preserve row count
        self.assertEqual(enc_df.shape[0], df.shape[0])

class TestArtifactsLoader(unittest.TestCase):
    def setUp(self):
        # Create a minimal TensorFlow + Keras stub so that artifacts.py can import it
        import types, sys
        tf_stub = types.ModuleType("tensorflow")
        keras_stub = types.ModuleType("tensorflow.keras")
        models_stub = types.ModuleType("tensorflow.keras.models")
        # default loader (will be monkeypatched inside test)
        def _dummy_loader(path): return object()
        models_stub.load_model = _dummy_loader
        keras_stub.models = models_stub
        class _DummyModel: pass
        keras_stub.Model = _DummyModel
        tf_stub.keras = keras_stub
        sys.modules.setdefault("tensorflow", tf_stub)
        sys.modules.setdefault("tensorflow.keras", keras_stub)
        sys.modules.setdefault("tensorflow.keras.models", models_stub)

    def test_load_artifacts_from_context_with_mocks(self):
        # We will mock tf.keras.models.load_model and provide temp pickle files
        import os
        import pickle
        import pandas as pd
        import utils.artifacts as artifacts_mod  # <-- updated to utils.artifacts
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmp:
            keras_model_path = os.path.join(tmp, "ae.h5")
            scalers_path = os.path.join(tmp, "scalers.pkl")
            year_gap_path = os.path.join(tmp, "year_gap.pkl")

            # Touch model path (actual loading is mocked)
            with open(keras_model_path, "wb") as f:
                f.write(b"dummy")

            # Prepare pickle files
            scalers_obj = {"ok": True}
            year_gap_obj = pd.DataFrame({"a": [1,2,3]})
            with open(scalers_path, "wb") as f:
                pickle.dump(scalers_obj, f)
            with open(year_gap_path, "wb") as f:
                pickle.dump(year_gap_obj, f)

            # Build a fake context
            class Ctx:
                artifacts = {
                    "keras_autoencoder_path": keras_model_path,
                    "scalers_path": scalers_path,
                    "year_gap_data_path": year_gap_path,
                }

            class DummyModel:
                def __init__(self, name="dummy"): self.name = name

            # Monkeypatch
            original_loader = tf.keras.models.load_model
            try:
                tf.keras.models.load_model = lambda p: DummyModel()
                ae = artifacts_mod.load_artifacts_from_context(Ctx)
            finally:
                tf.keras.models.load_model = original_loader

            # Assertions
            self.assertTrue(hasattr(ae, "model"))
            self.assertTrue(hasattr(ae, "scalers_dict"))
            self.assertTrue(hasattr(ae, "year_gap_data"))
            self.assertEqual(ae.scalers_dict["ok"], True)
            self.assertTrue("a" in ae.year_gap_data.columns)

if __name__ == "__main__":
    # Notebook/Databricks-safe: ignore kernel argv and avoid sys.exit()
    argv = ["first-arg-is-ignored"]
    unittest.main(argv=argv, exit=False, verbosity=2)
