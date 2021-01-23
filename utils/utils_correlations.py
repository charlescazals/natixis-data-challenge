"""Module contains functions of imputation with the correlation method """


from math import isnan
from datetime import timedelta
from typing import Union, Callable
from tqdm import tqdm
import numpy as np
import pandas as pd


def impute_df_with_correlations(
    dataframe: pd.DataFrame, corr_activation_f: Callable[[float], float]
) -> pd.DataFrame:
    """
    :param dataframe: the original data frame, with missing values
    :param corr_activation_f: calculates weights from correlations
    :return: the original dataframe, imputed using correlations and growth rates
    """
    print("Performing preliminary calculations...")
    growth_rates_df = get_growth_rates_df(dataframe)
    corr_df = get_correlations_df(dataframe)
    corr_df_activated = activate_correlations(corr_df, corr_activation_f)

    print("Imputing missing values...")
    df_imputed = fill_in_data_frame_correlations(dataframe, corr_df_activated, growth_rates_df)
    return df_imputed


# Growth rates computation

def get_growth_rates_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: the original data frame, with missing values
    :return: initial dataset growth rates (nan: infinite or unavailable)
    """
    growth_rates = dataframe.pct_change(fill_method=None)
    # --- change inf values to na (will be dropped later)
    growth_rates.replace([np.inf, -np.inf], np.nan, inplace=True)
    return growth_rates


# Correlations computation and activation

def get_correlations_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: the original data frame, with missing values
    :return: a correlation matrix of the original dataframe, based on pct changes
    """
    df_returns = get_growth_rates_df(dataframe)
    corr_df = df_returns.corr()
    # Set to zero when correlation cannot be computed (ex. constant time series)
    corr_df = corr_df.fillna(0)
    return corr_df


def activate_correlations(
    corr_df: pd.DataFrame, activation_f: Callable[[float], float]
) -> pd.DataFrame:
    """
    Activate a correlation data frame column by column, value by value, with pandas parallelism
    :param corr_df: a correlation matrix
    :param activation_f: a correlation pre processing function
    :return: a symmetric matrix of weights based on correlations
    """
    return corr_df.apply(lambda corr_series: corr_series.apply(activation_f), axis=0)


# Missing data interpolation using correlations and growth rates

def fill_in_data_frame_correlations(
    df_time_series: pd.DataFrame,
    corr_df_activated: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Impute missing value on the data frame, using correlations and growth rates
    :param df_time_series: initial dataset
    :param corr_df_activated: the correlation matrix after activation
    :param growth_rates_df: the initial dataset (pct changes)
    :returns: the dataset after imputation (where possible)
    """

    tqdm.pandas()  # show progress bar

    df_time_series_filled = df_time_series.progress_apply(
        lambda time_series: fill_in_time_series(
            df_time_series, time_series.name, corr_df_activated, growth_rates_df
        ),
        axis=0,
    )
    # Interpolate when correlation method fails to impute (ex. no correlated assets)
    df_final = df_time_series_filled.interpolate(
        method="linear", limit=None, limit_direction="forward"
    )

    # Compute and display some metrics
    num_accepted_nans = df_final.isna().sum().sum()
    num_values = df_final.count().sum()
    pct_originally_missing = (
        df_time_series.isna().sum().sum() - num_accepted_nans
    ) / num_values
    pct_missing_after_corr_method = (
        df_time_series_filled.isna().sum().sum() - num_accepted_nans
    ) / num_values
    print(f"% Missing Initially: {round(pct_originally_missing*100,1)}%")
    print(
    f"% Missing after imputation via correlations: {round(pct_missing_after_corr_method*100,1)}%"
    )
    print("% Missing after final interpolation: 0.0%")

    return df_final

def fill_in_time_series(
    df_time_series: pd.DataFrame,
    series_id: str,
    corr_df_activated: pd.DataFrame,
    growth_rates_df: pd.DataFrame,
) -> pd.Series:
    """
    Impute missing value on a time series, using correlations and growth rates
    """
    time_series_filled = pd.Series(
        index=df_time_series[series_id].index, name=series_id
    )
    not_started = True
    for timestamp_str, value in df_time_series[series_id].items():

        if isnan(value):
            if not_started:
                # data not yet available for this time series
                continue

            corr_series_activated = corr_df_activated[series_id]
            growth_rate_series = growth_rates_df.loc[timestamp_str]
            imputed_growth_rate = get_imputed_growth_rate(
                corr_series_activated, growth_rate_series
            )
            imputed_value = get_imputed_value(
                time_series_filled, timestamp_str, imputed_growth_rate
            )
            time_series_filled[timestamp_str] = imputed_value

        else:
            if not_started:
                not_started = False

            time_series_filled[timestamp_str] = value

    return time_series_filled


def get_imputed_value(
    time_series: pd.Series, timestamp_str: str, imputed_growth_rate: Union[float, None]
) -> float:
    """
    Returns an imputed value at a given time based on the last value and an imputed growth rate
    """
    if imputed_growth_rate is None:
        return np.nan

    prev_timestamp_str = get_previous_available_day(timestamp_str, time_series.index)
    prev_value = time_series[prev_timestamp_str]

    # Multiply by imputed growth rate
    imputed_value = prev_value * (1 + imputed_growth_rate)

    return imputed_value


def get_imputed_growth_rate(
    corr_series_activated: pd.Series, growth_rates: pd.Series
) -> Union[float, None]:
    """
    Returns an average of observed growth rates,
    weighted by pre processed correlations of the corresponding series
    """
    # Retrieve indices where growth rate is available
    not_nan_indices = np.where(np.logical_not(np.isnan(growth_rates)))[0]
    # Compute average of growth rates weighted by 'activated' correlation
    if np.sum(corr_series_activated[not_nan_indices]) == 0:
        return None
    imputed_growth_rate = np.average(
        growth_rates[not_nan_indices], weights=corr_series_activated[not_nan_indices]
    )
    return imputed_growth_rate


# Additional intermediary functions


def prev_day(timestamp_str: str, date_format: str = "%d/%m/%Y") -> str:
    """
    Retrieve the day-1 date in a given format
    """
    timestamp = pd.to_datetime(timestamp_str, format=date_format)
    prev_day_timestamp = timestamp + timedelta(days=-1)
    prev_day_timestamp_str = prev_day_timestamp.strftime(date_format)
    return prev_day_timestamp_str


def get_previous_available_day(timestamp_str: str, available_days: pd.Index) -> str:
    """
    Takes a current date string, returns the previous date string from possible of date strings.
    :param timestamp_str: the current date string
    :param available_days: the index of all date-strings with observations
    :return: the previous date string observed in the list
    """
    prev_timestamp_str = prev_day(timestamp_str)
    while prev_timestamp_str not in available_days:
        prev_timestamp_str = prev_day(prev_timestamp_str)
    return prev_timestamp_str


def truncate_below_threshold(num: float, threshold: float):
    """
    Truncates correlations below a certain threshold
    """
    if abs(num) < threshold:
        return 0
    return num
