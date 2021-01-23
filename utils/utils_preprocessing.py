"""Module contains preprocessing functions applied before imputation."""

import random
import numpy as np
import pandas as pd

def get_evaluation_set(dataframe: pd.DataFrame, method='linear', pct_nan=0.05):
    """Imputes missing data & remove data at random to evaluate imputation method.

    Args:
        dataframe: The input data.
        method:    How to interpolate the missing data (ffil, bfill, linear, spline).
        pct_nan:   % Missing values in evaluation set

    Returns:
        data_full: Dataframe with no missing data
        data_miss: Dataframe with observations removed (at random, outside of imputed observations)

    """
    # --- get location of native missing data
    mask = dataframe.isna()

    for val, idx in dataframe.apply(pd.Series.first_valid_index).items():
        mask.loc[idx, val] = True

    for val, idx in dataframe.apply(pd.Series.last_valid_index).items():
        mask.loc[idx, val] = True

    drop_idx = [
        (row, col)
        for row, col in zip(np.where(mask==False)[0], np.where(mask==False)[1])
    ]


    # --- impute the missing data
    random.seed(42)
    dataframe = dataframe.interpolate(
        method=method, limit=None, limit_direction="forward"
    )
    # --- drop observations at random
    dataframe_miss = dataframe.copy()

    for row, col in random.sample(drop_idx, int(round(pct_nan * len(drop_idx)))):
        dataframe_miss.iloc[row, col] = np.nan

    return dataframe, dataframe_miss
