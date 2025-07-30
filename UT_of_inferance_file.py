import unittest
import pandas as pd
import numpy as np

def onehot_df_to_index(df: pd.DataFrame) -> np.ndarray:
    return df.values.argmax(axis=1).astype("int32")[:, None]

def prepare_blocks(blocks, numeric_block_idx, embed_dim_rule=lambda k: max(2, int(np.ceil(np.sqrt(k))))):
    cat_arrays, cardinals, embed_dims = [], [], []
    for i, df in enumerate(blocks):
        if i == numeric_block_idx:
            num_array = df.values.astype("float32")
            continue
        arr = onehot_df_to_index(df)
        cat_arrays.append(arr)
        cardinals.append(df.shape[1])
        embed_dims.append(embed_dim_rule(df.shape[1]))
    return cat_arrays, num_array, cardinals, embed_dims

def remove_list(original_list, remove_items):
    return [item for item in original_list if item not in remove_items]

def face_value(df):
    if df['PrimaryCurr'] == df['BuyCurr']:
        df['FaceValue'] = abs(df['BuyAmount'])
    elif df['PrimaryCurr'] == df['SellCurr']:
        df['FaceValue'] = abs(df['SellAmount'])
    else:
        df['FaceValue'] = np.nan
    return df

def get_column_types(df):
    categorical_columns = [col for col in df.columns if df[col].dtype == object]
    numeric_columns = [col for col in df.columns if df[col].dtype in (int, float)]
    return categorical_columns, numeric_columns

class TestInferenceUtils(unittest.TestCase):

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

    def test_remove_list(self):
        original = ['a', 'b', 'c']
        to_remove = ['b']
        result = remove_list(original, to_remove)
        self.assertListEqual(result, ['a', 'c'])

    def test_face_value_buycurr(self):
        row = {
            'PrimaryCurr': 'USD',
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'BuyAmount': 1000.0,
            'SellAmount': 900.0
        }
        df = pd.Series(row)
        result = face_value(df)
        self.assertEqual(result['FaceValue'], 1000.0)

    def test_face_value_sellcurr(self):
        row = {
            'PrimaryCurr': 'EUR',
            'BuyCurr': 'USD',
            'SellCurr': 'EUR',
            'BuyAmount': 1000.0,
            'SellAmount': 900.0
        }
        df = pd.Series(row)
        result = face_value(df)
        self.assertEqual(result['FaceValue'], 900.0)

    def test_get_column_types(self):
        df = pd.DataFrame({
            'A': ['a', 'b'],
            'B': [1.0, 2.0],
            'C': [3, 4]
        })
        cats, nums = get_column_types(df)
        self.assertIn('A', cats)
        self.assertIn('B', nums)
        self.assertIn('C', nums)

if __name__ == "__main__":
    unittest.main()
