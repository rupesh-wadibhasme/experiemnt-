import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import inference_script as script  # replace with actual filename/module name


class TestInferenceScript(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame([{
            'Instrument': 'FXSPOT',
            'BUnit': 'BU1',
            'Cpty': 'CP1',
            'ActivationDate': datetime(2025, 7, 1),
            'MaturityDate': datetime(2025, 7, 10),
            'PrimaryCurr': 'USD',
            'BuyAmount': 1000.0,
            'SellAmount': 950.0,
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'SpotRate': 1.1,
            'ForwardPoints': 0.02
        }])

        self.year_gap_df = pd.DataFrame([{
            'BUnit': 'BU1',
            'Cpty': 'CP1',
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'ActivationDate': datetime(2025, 6, 30)
        }])

    @patch('inference_script.keras.models.load_model')
    @patch('inference_script.pickle.load')
    def test_get_model_returns_model_and_scalers(self, mock_pickle, mock_load_model):
        dummy_model = MagicMock()
        dummy_scalers = {'IXOM': 'dummy_scaler'}
        mock_load_model.return_value = dummy_model
        mock_pickle.return_value = dummy_scalers

        model, scalers = script.get_model('rcc')

        self.assertEqual(model, dummy_model)
        self.assertEqual(scalers, 'dummy_scaler')

    def test_get_column_types(self):
        df = pd.DataFrame({
            'col1': ['A', 'B'],
            'col2': [1.0, 2.0],
            'col3': [10, 20]
        })
        cat, num = script.get_column_types(df)
        self.assertIn('col1', cat)
        self.assertIn('col2', num)
        self.assertIn('col3', num)

    def test_face_value_buycurr(self):
        row = pd.Series({
            'PrimaryCurr': 'USD',
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'BuyAmount': 100,
            'SellAmount': 90
        })
        result = script.face_value(row)
        self.assertEqual(result['FaceValue'], 100)

    def test_face_value_sellcurr(self):
        row = pd.Series({
            'PrimaryCurr': 'EUR',
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'BuyAmount': 100,
            'SellAmount': 90
        })
        result = script.face_value(row)
        self.assertEqual(result['FaceValue'], 90)

    def test_remove_list(self):
        original = ['A', 'B', 'C']
        to_remove = ['B']
        result = script.remove_list(original, to_remove)
        self.assertNotIn('B', result)

    def test_check_missing_group_detects_missing(self):
        unique_BU, unique_cpty, unique_pc = set(), set(), set()
        data = pd.DataFrame([{
            'BUnit': 'BUX',
            'Cpty': 'CPX',
            'PrimaryCurr': 'XYZ'
        }])
        flag, msg = script.check_missing_group(unique_BU, unique_cpty, unique_pc, data)
        self.assertTrue(flag)
        self.assertIn("has not previously engaged", msg)

    def test_check_currency_new_pair(self):
        trained = {
            'CP1': {'buy': ['USD'], 'sell': ['EUR']}
        }
        df = pd.DataFrame([{
            'Cpty': 'CP1',
            'BuyCurr': 'INR',
            'SellCurr': 'JPY'
        }])
        flag, msg = script.check_currency(df, trained)
        self.assertTrue(flag)
        self.assertIn('has not previously', msg)

    def test_get_condition_filters(self):
        df = pd.DataFrame([{
            'BUnit_A': 0.2,
            'Cpty_B': 0.3,
            'BuyCurr_USD': 0.96,
            'SellCurr_EUR': 0.99,
            'TDays': 0.6
        }])
        anomaly, normal, features = script.get_condition_filters(df)
        self.assertEqual(anomaly, 1)
        self.assertEqual(normal, 0)
        self.assertIsInstance(features, list)

    def test_get_filtered_data_structure(self):
        features = pd.DataFrame([{
            'BUnit_BU1': 1, 'Cpty_CP1': 1, 'BuyCurr_USD': 0.5, 'SellCurr_EUR': 0.6
        }])
        reconstructed_df = pd.DataFrame([{
            'BUnit_BU1': 0.9, 'Cpty_CP1': 0.8, 'BuyCurr_USD': 0.3, 'SellCurr_EUR': 0.2
        }])
        result = script.get_filtered_data(features, reconstructed_df)
        self.assertIn('BUnit', result)
        self.assertIn('Cpty', result)
        self.assertIn('Deviated_Features', result)

    @patch('inference_script.inference')
    def test_anomaly_prediction_business_rule(self, mock_inference):
        mock_inference.return_value = ("Deal appears anomalous due to rule", '', '', '', '')
        input_data = {
            'Client': 'rcc',
            'UniqueId': '123',
            'Instrument': 'FXSPOT',
            'BUnit': 'BU1',
            'Cpty': 'CP1',
            'ActivationDate': datetime(2025, 7, 1),
            'MaturityDate': datetime(2025, 7, 10),
            'PrimaryCurr': 'USD',
            'BuyAmount': 1000,
            'SellAmount': 950,
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'SpotRate': 1.1,
            'ForwardPoints': 0.01
        }

        with patch("inference_script.pickle.load", return_value=self.year_gap_df):
            result = script.anomaly_prediction(input_data)
            self.assertEqual(result['Anomaly'], 'Y')
            self.assertIn('Deal appears anomalous', result['Reason'])

