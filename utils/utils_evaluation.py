"""Module contains functions to evaluate imputation methods"""

import numpy as np
import pandas as pd

def eval_imputation(df_full: pd.DataFrame, df_pred: pd.DataFrame, df_miss: pd.DataFrame):
    """Computes nrmse/frobenius-norm between df_full & df_pred.

    Args:
        df_full:  The target data.
        df_pred: The prediction data.
        df_miss: The data before interpolation

    Returns:
        nrmse_dict: Dictionary {time series: (nrmse, #NaNs)}
        nrmse: Average normalized root mean squared error
        fnorm: Frobenius norm ||df_full-df_pred||F

    """
    # --- check nans match in target/prediction
    assert df_full.isna().equals(df_pred.isna())
    diff = df_full - df_pred

    # --- nrmse
    nrmse = [
        np.sqrt((diff[col] ** 2).sum()) / (np.finfo(float).eps + df_full[col].std())
        for col in df_full.columns
    ]
    nrmse_dict = {df_full.columns[i]: (nrmse[i],
                                       df_miss.isna().sum()[i]-df_full.isna().sum()[i])
                  for i in range(len(nrmse))}

    # --- frobenius norm
    fnorm = np.linalg.norm(diff.fillna(0), "fro")

    return nrmse_dict, np.mean(nrmse), fnorm
