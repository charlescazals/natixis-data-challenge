"""Module contains functions to preprocess the data and perform imputations using xgboost"""
import numpy as np
import pandas as pd
import xgboost as xgb

from utils_correlations import get_growth_rates_df
from utils_correlations import get_correlations_df
from utils_correlations import activate_correlations
from utils_correlation_activations import mixed_truncate_inverse_distance

def impute_dataframe_xgboost(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing data via a regression-v=based model.
    Note: linear interpollation when model cannot be used

    Args:
        dataframe: The input data.

    Returns:
        data_full: Dataframe with no missing data

    """
    feature_map = get_feature_map(dataframe)
    dataframe = fill_in_data_frame_xgboost(dataframe, feature_map, verbose=False)
    return dataframe.interpolate(method='linear', limit=None, limit_direction='forward')

def get_lags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates lagged features in dataset (±1 day).

    Args:
        dataframe: The input data.

    Returns:
        dataframe: Dataframe with lagged columns

    """
    for lag in dataframe.columns:
        dataframe.loc[:, "lag_-1_" + str(lag)] = dataframe[lag].shift(1)
        dataframe.loc[:, "lag_+1_" + str(lag)] = dataframe[lag].shift(-1)
        dataframe.loc[:, "mean-" + str(lag)] = (
            dataframe["lag_+1_" + str(lag)] + dataframe["lag_-1_" + str(lag)]
        ) * 0.5
    return dataframe

def get_feature_map(dataframe: pd.DataFrame) -> dict:
    """Selects the most column time series to each column in dataset.

    Args:
        dataframe: The input data.

    Returns:
        feature_map: feature map {'series': [correlated series]}

    """
    # --- calculate growth rates and correlations
    df_growth_rates = get_growth_rates_df(dataframe)
    corr_df = get_correlations_df(df_growth_rates)
    corr_df = activate_correlations(corr_df, mixed_truncate_inverse_distance)
    # --- compile feature map
    feature_map = {}
    for i, row in dataframe.iterrows():
        features = []
        for col in row.index:
            if row[col] < 1 and abs(row[col]) > 0.9:
                features.append(col)
        feature_map[i] = features
    return feature_map


def get_series_with_imputations(
    dataframe: pd.DataFrame, series: str, dataframe_full: pd.DataFrame
) -> np.ndarray:
    """Trains xgboost model and imputes missing data for a given series.

    Args:
        dataframe: the input data.
        series: the series to impute
        dataframe_full : the reference dataframe (training)

    Returns:
        array: imputed series values

    """
    u_test = dataframe[dataframe[series].isna()]
    u_train = dataframe.dropna(how='all', subset=[series])
    y_train = u_train[series]
    x_train = u_train.drop([series], axis=1)
    x_test = u_test.drop([series], axis=1)

    regressor = xgb.XGBRegressor()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    x_test.loc[:,'pred'] = y_pred

    exp = dataframe_full.copy()

    for i,_ in x_test.iterrows():
        exp.loc[i, series] = x_test.loc[i, 'pred']

    return exp[series].values

def model_for_series(series: str , feats: list, dataframe_full: pd.DataFrame) -> np.ndarray:
    """Calls the preprocessing and the prediction function for a single series

    Args:
        series: the series to impute.
        features: the features to use.
        dataframe_full : the reference dataframe (training)

    Returns:
        array: imputed series values

    """
    if len(feats) == 0:
        return dataframe_full[series].values

    dataframe_useful = dataframe_full.loc[:, feats]
    dataframe_useful.loc[:, series] = dataframe_full.loc[:, series]
    first_valid_index = dataframe_useful.loc[:,series].first_valid_index()
    dataframe_truncated = get_lags(dataframe_useful.loc[first_valid_index:,:])

    return get_series_with_imputations(dataframe_truncated, series, dataframe_full)

def fill_in_data_frame_xgboost(
    dataframe: pd.DataFrame, feature_map: dict, verbose=False
) -> pd.DataFrame:
    """Imputes dataframe via regression-based model

    Args:
        dataframe: the dataframe to impute.
        feature_map: the features to features map to use for prediction.
        verbose:   Controls verbosity when imputing time series

    Returns:
        dataframe: the imputed dataset (where possible)

    """
    for feat in feature_map.keys():
        if verbose:
            print('imputing', feat)
        k = k + 1
        dataframe.loc[:,feat] = model_for_series(feat, feature_map[feat], dataframe)
    return dataframe
