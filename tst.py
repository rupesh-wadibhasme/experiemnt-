# unit_tests_simple.py
# Minimal, single-file unittest suite (lighter coverage, no TF/fx_serving stubs)
# Run:
#   python unit_tests_simple.py
# Or in a notebook:
#   %run unit_tests_simple.py

import os, sys, unittest, json, importlib
import pandas as pd

# Ensure project root on path (this file's dir)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from utils package (make sure utils/__init__.py exists)
serializers = importlib.import_module("utils.serializers")
rules = importlib.import_module("utils.rules")
preprocess = importlib.import_module("utils.preprocess")

class TestSerializers(unittest.TestCase):
    def test_to_utc(self):
        from datetime import datetime, timezone
        self.assertEqual(int(serializers.to_utc(0).timestamp()), 0)
        self.assertEqual(int(serializers.to_utc(1.5).timestamp()), 1)
        self.assertEqual(int(serializers.to_utc("1970-01-01T00:00:01Z").timestamp()), 1)
        naive = datetime(1970,1,1,0,0,2)
        self.assertEqual(int(serializers.to_utc(naive).timestamp()), 2)
        aware = datetime(1970,1,1,0,0,3, tzinfo=timezone.utc)
        self.assertIs(serializers.to_utc(aware), aware)
        self.assertIsNone(serializers.to_utc(""))
        self.assertIsNone(serializers.to_utc(object()))

    def test_to_json_safely(self):
        out = serializers.to_json_safely({"a": 1})
        self.assertEqual(json.loads(out), {"a": 1})
        df = pd.DataFrame([{"x":1},{"x":2}])
        self.assertEqual(json.loads(serializers.to_json_safely(df)), [{"x":1},{"x":2}])
        class X: pass
        s = serializers.to_json_safely(X())
        self.assertIsInstance(s, str); json.loads(s)

class TestRules(unittest.TestCase):
    def test_compare_rows(self):
        df1 = pd.DataFrame([
            {"BUnit":"A","Cpty":"C1","BuyCurr":"USD","SellCurr":"INR","DealDate":"2024-01-01"},
            {"BUnit":"B","Cpty":"C2","BuyCurr":"EUR","SellCurr":"USD","DealDate":"2024-02-01"},
        ])
        df2_match = pd.DataFrame([{"BUnit":"B","Cpty":"C2","BuyCurr":"EUR","SellCurr":"USD","DealDate":"2024-02-01"}])
        df2_nomatch = pd.DataFrame([{"BUnit":"X","Cpty":"C9","BuyCurr":"JPY","SellCurr":"GBP","DealDate":"2024-03-01"}])
        self.assertTrue(rules.compare_rows(df1, df2_match))
        self.assertFalse(rules.compare_rows(df1, df2_nomatch))

    def test_check_missing_group(self):
        data = pd.DataFrame([{"BUnit":"A","Cpty":"C1","PrimaryCurr":"USD"}])
        scalers = {}
        code,_ = rules.check_missing_group(data, scalers, set(), set(), set()); self.assertEqual(code,1)
        scalers = {"grouped_scalers": {("B","C2","EUR"): {}}}
        code,_ = rules.check_missing_group(data, scalers, {"B"},{"C2"},{"EUR"}); self.assertEqual(code,1)
        scalers = {"grouped_scalers": {("A","C1","USD"): {}}}
        code,_ = rules.check_missing_group(data, scalers, set(),{"C1"},{"USD"}); self.assertEqual(code,1)
        code,_ = rules.check_missing_group(data, scalers, {"A"},{"C1"},{"USD"}); self.assertEqual(code,0)

    def test_check_currency(self):
        data = pd.DataFrame([{"Cpty":"C1","BuyCurr":"USD","SellCurr":"INR"}])
        scalers = {"cpty_group": {}}
        self.assertEqual(rules.check_currency(data, scalers)[0], 1)
        scalers = {"cpty_group": {"C1":{"buy":{"EUR":{}}, "sell":{"GBP":{}}}}}
        self.assertEqual(rules.check_currency(data, scalers)[0], 1)
        scalers = {"cpty_group": {"C1":{"buy":{"EUR":{}}, "sell":{"INR":{}}}}}
        self.assertEqual(rules.check_currency(data, scalers)[0], 1)
        scalers = {"cpty_group": {"C1":{"buy":{"USD":{}}, "sell":{"INR":{}}}}}
        self.assertEqual(rules.check_currency(data, scalers)[0], 0)

class TestPreprocess(unittest.TestCase):
    def test_get_column_types_and_one_hot(self):
        df = pd.DataFrame({"cat":["x","y","z"], "num1":[1.0,2.0,3.0], "num2":[1,2,3]})
        cats, nums = preprocess._get_column_types(df)
        self.assertIn("cat", cats); self.assertIn("num1", nums); self.assertIn("num2", nums)

        # OneHotEncoder (no special options; simple)
        from sklearn.preprocessing import OneHotEncoder
        colors = pd.DataFrame({"color": ["red","blue","red","green"]})
        enc = OneHotEncoder(sparse=False).fit(colors)
        enc_df = preprocess._one_hot(colors, enc)
        self.assertEqual(enc_df.shape, (4, 3))

if __name__ == "__main__":
    # Manual suite runner (notebook/Databricks friendly; no argv parsing)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2).run(suite)
