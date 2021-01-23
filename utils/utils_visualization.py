"""Module contains visualization functions to inspect input data and results
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(dataframe: pd.DataFrame, category: str=None, show_corr: bool=False) -> None:
    """Visualization function to display missing values and correlations in a given dataset.

    Args:
        dataframe: The input data.
        category:  The filter to apply to the series type.
        show_corr: Whether to plot the corrrelation (pearson)matrix.

    Returns:
        None

    """
    if category:
        dataframe = dataframe.loc[:, dataframe.columns.str.contains(category)]

    if show_corr:
        _ , axs = plt.subplots(1, 2, figsize=(30, 15))
        # --- display missing values
        sns.heatmap(dataframe.isna(),
                    cbar=False,
                    ax=axs[0])
        # --- display correlation heatmap
        corr = dataframe.corr()
        sns.heatmap(corr,
                    mask = np.triu(np.ones_like(corr, dtype=bool)),
                    ax = axs[1],
                    center=0)
    else:
        _ , axs = plt.subplots(1, 1, figsize=(15,15))
        # --- display missing values
        sns.heatmap(dataframe.isna(),
                    cbar=False,
                    ax=axs)
    return 0


def plot_compare(results:dict, filename: str):
    """Visualization function to display missing values and correlations in a given dataset.

    Args:
        results: Dictionary {model: results_model}
            results_model should match output from eval_imputation
        log_y:   Display y-axis in log-scale (useful for large nrmses)

    Returns:
        None

    """
    df_res=pd.DataFrame()

    for res,method in zip(results.values(), results.keys()):
        df_temp = pd.DataFrame(res[0], index=["nrmse", "nan"]).transpose()
        df_temp["type"] = [s[0] for s in df_temp.index.str.split("_")]
        df_temp["method"] = method

        df_res = pd.concat([df_res, df_temp], axis=0)

    fig , axs = plt.subplots(1, 1, figsize=(30, 15))
    axs = sns.boxplot(x="type", y="nrmse", hue="method", data=df_res)

    axs.set_ylim([0, 10])
    axs.set_xlabel("Series Type", fontsize=30)
    axs.set_xticklabels(axs.get_xmajorticklabels(), fontsize=30)
    axs.set_ylabel("NRMSE", fontsize=30)

    axs.legend(loc="upper left", fontsize=25)
    fig.savefig(f'./results/img/{filename}.png')
    return 0

def plot_feature_map(feature_map):
    _ , axs = plt.subplots(1, 1, figsize=(30, 15))
    
    # Histogram of nb of feature per model
    nb_feature = [len(x) for x in feature_map.values()]
    n, bins, patches = plt.hist(
        x=nb_feature, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("number of features for regression")
    plt.ylabel("Frequency")
    plt.title("number of features for regression")
    plt.text(23, 45, r"$\mu=15, b=3$")
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    return 0