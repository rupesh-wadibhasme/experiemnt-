import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from embed_ad_infer import *

class TestInferenceFullCoverage(unittest.TestCase):
    def test_onehot_df_to_index(self):
        df = pd.DataFrame([[0, 1, 0], [1, 0, 0]])
        expected = np.array([[1], [0]])
        result = onehot_df_to_index(df)
        np.testing.assert_array_equal(result, expected)

    def test_prepare_blocks(self):
        df1 = pd.DataFrame([[0, 1, 0], [1, 0, 0]])
        df2 = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]])
        cat, num, cards, embeds = prepare_blocks([df1, df2], numeric_block_idx=1)
        self.assertEqual(len(cat), 1)
        self.assertEqual(num.shape, (2, 2))
        self.assertEqual(cards, [3])

    def test_get_column_types(self):
        df = pd.DataFrame({'A': ['x', 'y'], 'B': [1.0, 2.0], 'C': [1, 2]})
        cats, nums = get_column_types(df)
        self.assertEqual(set(cats), {'A'})
        self.assertEqual(set(nums), {'B', 'C'})

    def test_one_hot(self):
        df = pd.DataFrame({'Color': ['Red', 'Blue', 'Red']})
        encoded, _ = one_hot(df)
        self.assertTrue("Color_Red" in encoded.columns)

    def test_scale(self):
        df = pd.DataFrame({'Val': [10, 20, 30]})
        scaled, _ = scale(df)
        self.assertTrue((scaled['Val'] <= 1).all())

    def test_remove_list(self):
        self.assertEqual(remove_list(['a', 'b', 'c'], ['b']), ['a', 'c'])

    def test_face_value_buy(self):
        row = pd.Series({
            'PrimaryCurr': 'USD', 'BuyCurr': 'USD', 'SellCurr': 'EUR',
            'BuyAmount': 100, 'SellAmount': 80
        })
        result = face_value(row)
        self.assertEqual(result['FaceValue'], 100)

    def test_face_value_sell(self):
        row = pd.Series({
            'PrimaryCurr': 'EUR', 'BuyCurr': 'USD', 'SellCurr': 'EUR',
            'BuyAmount': 100, 'SellAmount': 80
        })
        result = face_value(row)
        self.assertEqual(result['FaceValue'], 80)

    def test_get_uniques(self):
        d = {('B1', 'C1', 'USD'): {}, ('B2', 'C2', 'EUR'): {}}
        bu, cp, pc = get_uniques(d)
        self.assertIn('B1', bu)
        self.assertIn('C2', cp)
        self.assertIn('USD', pc)

    def test_check_missing_group(self):
        d = {('B1', 'C1', 'USD'): {}}
        df = pd.DataFrame([{'BUnit': 'B2', 'Cpty': 'C2', 'PrimaryCurr': 'EUR'}])
        bu, cp, pc = get_uniques(d)
        flag, msg = check_missing_group(bu, cp, pc, df)
        self.assertEqual(flag, 1)

    def test_check_currency(self):
        df = pd.DataFrame([{'Cpty': 'C1', 'BuyCurr': 'USD', 'SellCurr': 'EUR'}])
        scalers = {'C1': {'buy': ['INR'], 'sell': ['GBP']}}
        flag, msg = check_currency(df, scalers)
        self.assertEqual(flag, 1)

    def test_compare_rows(self):
        today = datetime.today()
        df1 = pd.DataFrame([{'BUnit': 'B1', 'Cpty': 'C1', 'BuyCurr': 'USD', 'SellCurr': 'EUR', 'ActivationDate': today}])
        df2 = df1.copy()
        self.assertTrue(compare_rows(df1, df2))

    @patch("embed_ad_infer.get_model")
    def test_inference_year_gap_returns_message(self, mock_get_model):
        df = pd.DataFrame([{
            'Instrument': 'FXSPOT', 'BUnit': 'B1', 'Cpty': 'C1', 'PrimaryCurr': 'USD',
            'BuyCurr': 'USD', 'SellCurr': 'EUR', 'BuyAmount': 1000.0, 'SellAmount': 950.0,
            'SpotRate': 1.2, 'ForwardPoints': 0.05,
            'ActivationDate': datetime.today(),
            'MaturityDate': datetime.today() + timedelta(days=1),
        }])
        year_gap = df.copy()
        result = inference(df, year_gap)
        self.assertIsInstance(result[0], str)

if __name__ == "__main__":
    unittest.main()
