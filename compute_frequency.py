import pandas as pd
from collections import Counter
import pickle

def compute_frequency_stats(df: pd.DataFrame, output_path='frequency_stats.pkl'):
    """
    Compute frequency of key categorical combinations for later use in anomaly reasoning.
    Args:
        df (pd.DataFrame): Cleaned training dataframe
        output_path (str): File path to store the stats pickle
    """
    freq_dict = {}

    # Example combinations
    combos_to_count = [        
        ('BUnit', 'Cpty', 'BuyCurr','SellCurr') 
    ]

    for combo in combos_to_count:
        key_counts = Counter(tuple(row) for row in df[list(combo)].values)
        freq_dict.update(key_counts)

    # Save to pickle
    with open(output_path, 'wb') as f:
        pickle.dump(freq_dict, f)

    print(f"[✓] Frequency stats saved to: {output_path}")
