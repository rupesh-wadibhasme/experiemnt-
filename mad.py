import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_mad_thresholds(df, transtype_col='Transtype', value_col='Tdays', c=3, visualize=False):
    """
    Computes MAD-based thresholds for subgroups in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input data.
    transtype_col (str): The column name for transaction type.
    value_col (str): The column name with numerical values.
    c (float): Multiplier for MAD (default is 3).
    visualize (bool): Whether to plot distributions with thresholds.

    Returns:
    pd.DataFrame: A DataFrame with Transtype and calculated threshold.
    """

    def calculate_mad_threshold(series):
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        return median + c * mad

    thresholds = df.groupby(transtype_col)[value_col].apply(calculate_mad_threshold).reset_index()
    thresholds.columns = [transtype_col, 'MAD_Threshold']

    if visualize:
        unique_types = df[transtype_col].unique()
        for ttype in unique_types:
            subset = df[df[transtype_col] == ttype][value_col]
            threshold = thresholds.loc[thresholds[transtype_col] == ttype, 'MAD_Threshold'].values[0]
            plt.figure()
            plt.hist(subset, bins=30, alpha=0.7)
            plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
            plt.title(f'{ttype} - Tdays Distribution with MAD Threshold')
            plt.xlabel(value_col)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.show()

    return thresholds
