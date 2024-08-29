import pandas as pd
import numpy as np
import argparse

from typing import Tuple
from matplotlib import pyplot as plt
from experiments.util import collect_exp_results, ucb, lcb, median, count
import math


def different_method_plot(df_agg: pd.DataFrame, metric: str = 'nll', display: bool = True,
                          filter_std_higher_than: float = math.inf) \
        -> Tuple[pd.DataFrame, plt.Figure]:
    models = set(df_agg['model'])

    best_rows = []
    for model in models:
        df_agg_model = df_agg.loc[df_agg['model'] == model]
        df_agg_model = df_agg_model[df_agg_model[(metric, 'std')] < filter_std_higher_than]
        df_agg_model.sort_values(by=(metric, 'mean'), ascending=True, inplace=True)
        best_rows.append(df_agg_model.iloc[0])

    best_rows_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots()
    x_pos = 2 * np.arange(len(models))
    ax.bar(x_pos, best_rows_df[(metric, 'mean')], color='#ADD8E6')
    ax.errorbar(x_pos, best_rows_df[(metric, 'mean')], yerr=best_rows_df[(metric, 'std')],
                linestyle='', capsize=4., color='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(best_rows_df['model'], rotation=-20)
    ax.set_ylabel(metric)
    fig.tight_layout()

    if display:
        plt.show()
        print(best_rows_df[[('model', ''), ('nll', 'mean'), ('nll', 'std'),
                            ('rmse', 'mean'), ('rmse', 'std')]].to_string())

    return best_rows_df, fig


def main(args, drop_nan=False):
    df_full, param_names = collect_exp_results(exp_name=args.exp_name)
    df_full = df_full[df_full['data_source'] == args.data_source]

    for col in ['bandwidth_kde', 'bandwidth_ssge', 'bandwidth_score_estim']:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(value='auto')

    if drop_nan:
        result_is_nan = df_full['nll'].isna()
        df_full_dropped = df_full[result_is_nan]
        print(f'Dropped results due to NaNs: {df_full_dropped.shape[0]}')
        df_full = df_full[~result_is_nan]
    else:
        # replace nan with bad values
        for col, noise_std in [('nll', 1000.), ('rmse', 5.)]:
            series = df_full[col]
            max_value = series.max()
            is_nan = series.isna()
            noise = np.random.normal(0, noise_std, size=is_nan.sum())  # Adjust mean and stddev as needed
            series[is_nan] = max_value + noise
            df_full[col] = series

    # group over everything except seeds and aggregate over the seeds
    groupby_names = list(set(param_names) - {'model_seed', 'data_seed'})

    # replace all the nans in hyperparameter columns with 'N/A'
    for column in groupby_names:
        df_full[column] = df_full[column].fillna('N/A')

    groupby_names.remove('likelihood_std')
    # groupby_names.remove('added_gp_outputscale')

    # replace the added_gp_outputscale with the first element of the list
    if not type(df_full['added_gp_outputscale'].iloc[0]) == np.float_:

        df_full['added_gp_outputscale'] = df_full['added_gp_outputscale'].apply(lambda x: x[0])

    # first take mean over the data seeds
    df_mean = df_full.groupby(by=groupby_names + ['model_seed'], axis=0).mean()
    df_mean.reset_index(drop=False, inplace=True)

    # then compute the stats over the model seeds
    df_agg = df_mean.groupby(by=groupby_names, axis=0).aggregate(['mean', 'std', count], axis=0)# ucb, lcb, median, count], axis=0)
    df_agg.reset_index(drop=False, inplace=True)
    # df_agg.sort_values(by=[('nll', 'mean')], ascending=True, inplace=True)

    # filter all the rows where the count is less than 3
    df_agg = df_agg[df_agg['rmse']['count'] >= 3]

    # print('Available models:', set(df_agg['model']))
    # print('Best nll', df_agg[('nll', 'mean')][0] )

    print('Models:', set(df_agg['model']))

    different_method_plot(df_agg, metric='nll')
    different_method_plot(df_agg, metric='rmse')

    df_method = df_agg[(df_agg['model'] == 'BNN_FSVGD_SimPrior_gp')]

    #df_method = df_method[df_method['bandwidth_score_estim'] > 0.1]

    metric = 'nll'
    for param in ['num_measurement_points', 'added_gp_lengthscale', 'added_gp_outputscale']:# ['num_f_samples', 'bandwidth_score_estim', 'bandwidth_svgd', 'num_measurement_points']:
        plt.scatter(df_method[param], df_method[(metric, 'mean')])
        plt.xlabel(param)
        #plt.xscale('log')
        plt.ylabel(metric)
        # plt.scatter(1.5, -3.5, color='red')
        # plt.ylim(-3.6, -3.)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect results of a regression experiment.')
    parser.add_argument('--exp_name', type=str, default='test_aug28')
    parser.add_argument('--data_source', type=str, default='spot_real')
    args = parser.parse_args()
    main(args)