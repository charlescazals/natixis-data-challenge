"""
contains the main function to operate the repository's core features
"""
import os
import sys
import pickle as pkl

import pandas as pd

from utils.utils_preprocessing import get_evaluation_set
from utils.utils_correlations import impute_df_with_correlations
from utils.utils_correlation_activations import mixed_truncate_inverse_distance
from utils.utils_xgboost import impute_dataframe_xgboost
from utils.utils_evaluation import eval_imputation
from utils.utils_visualization import plot_compare

def main(args):
    """Main function
    """
    # --- load initial dataset & mapping
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_NAME), index_col=0)
    mapping = pd.read_csv(os.path.join(DATA_DIR, MAPPING_NAME))
    # --- apply mapping
    df.columns = [str(typ) + "_" + str(col) for col, typ in zip(df.columns, mapping.Type)]

    if len(args) > 0 :
        if args[0] == 'impute':
            if args[1] == 'correlations':
                print('imputing dataset using correlation-based model')
                df_imputed = impute_df_with_correlations(df, CORR_ACTIVATION_F)
                df_imputed.to_csv('results/imputed_data.csv')

            elif args[1] == 'xgboost':
                print('imputing dataset using regression-based (xgboost) model...')
                df_imputed = impute_dataframe_xgboost(df)
                df_imputed.to_csv('results/imputed_data.csv')

            else:
                print('imputing dataset using both model')
                # correlation-based
                df_imputed = impute_df_with_correlations(df, CORR_ACTIVATION_F)
                df_imputed.to_csv('results/imputed_data.csv')
                # xgboost
                df_imputed = impute_dataframe_xgboost(df)
                df_imputed.to_csv('results/imputed_data.csv')

        elif args[0] == 'evaluate':
            # --- create evaluation set
            df_full, df_miss = get_evaluation_set(
                df.reset_index().drop("Date", axis=1), method="linear"
            )
            if args[1] == 'correlations':
                print('evaluating correlation-based model')
                # --- impute missing data
                df_pred_baseline = df_miss.interpolate(
                    method="linear", limit=None, limit_direction="forward"
                )
                df_pred_corr = impute_df_with_correlations(
                    df_miss.set_index(df.index), CORR_ACTIVATION_F
                )
                # --- evaluate models
                results_dict = {
                    "baseline": eval_imputation(df_full, df_pred_baseline, df_miss),
                    "correlations": eval_imputation(
                        df_full, df_pred_corr.reset_index().drop("Date", axis=1), df_miss
                    ),
                }
                # save results to pkl file (TODO: create .ipynb to postprocess results)
                with open('./results/eval_corr.pkl', 'wb') as handle:
                    pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                # --- plot results and save to ./img/results
                plot_compare(results=results_dict, filename='eval_corr')
            elif args[1] == 'xgboost':
                print('evaluating xgboost-based model, this may take a while...')
                # --- impute missing data
                df_pred_baseline = df_miss.interpolate(
                    method="linear", limit=None, limit_direction="forward"
                )
                df_pred_xgb = impute_dataframe_xgboost(df)
                # --- evaluate models
                results_dict = {
                    "baseline": eval_imputation(df_full, df_pred_baseline, df_miss),
                    "xgboost": eval_imputation(
                        df_full, df_pred_xgb.reset_index().drop("Date", axis=1), df_miss
                    ),
                }
                # save results to pkl file (TODO: create .ipynb to postprocess results)
                with open('./results/eval_xgb.pkl', 'wb') as handle:
                    pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                # --- plot results and save to ./img/results
                plot_compare(results=results_dict, filename='eval_xgb')
            else:
                print('evaluating both model...')
                # --- impute missing data
                df_pred_baseline = df_miss.interpolate(
                    method="linear", limit=None, limit_direction="forward"
                )
                df_pred_corr = impute_df_with_correlations(
                    df_miss.set_index(df.index), CORR_ACTIVATION_F
                )
                df_pred_xgb = impute_dataframe_xgboost(df)
                # --- evaluate models
                results_dict = {
                    "baseline": eval_imputation(df_full, df_pred_baseline, df_miss),
                    "correlations": eval_imputation(
                        df_full, df_pred_corr.reset_index().drop("Date", axis=1), df_miss
                    ),
                    "xgboost": eval_imputation(
                        df_full, df_pred_xgb.reset_index().drop("Date", axis=1), df_miss
                    ),
                }
                # save results to pkl file (TODO: create .ipynb to postprocess results)
                with open('./results/eval_ALL.pkl', 'wb') as handle:
                    pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
                # --- plot results and save to ./img/results
                plot_compare(
                    results=results_dict, filename='eval_ALL'
                )
                sys.exit(0)

    else:
        print('Invalid input, please refer to README.md')
        sys.exit(0)

def validate_args(args):
    """
    Checks that the argument entered in the command line are correct otherwise it breaks

    Parameters
    ----------
    args : list
        args entered in the commande line without the first element : sys.argv[1:]

    """
    #Getting and validating args from command line
    if args[0] not in ['impute','evaluate']:
        print('invalid args: please choose an action in < impute , evaluate >')
        sys.exit(0)

if __name__ == '__main__':
    DATA_DIR = "data/raw_in/"
    DATA_OUT = 'data/data_processed/'
    FILE_NAME = "Risques 2/data_set_challenge.csv"
    MAPPING_NAME = "Risques 2/final_mapping_candidat.csv"

    CORR_ACTIVATION_F = mixed_truncate_inverse_distance

    # --- load data
    os.system('python3 ./utils/setup.py')

    validate_args(sys.argv[1:])
    main(sys.argv[1:])
