from experiments.util import (generate_run_commands, generate_base_command, RESULT_DIR, sample_param_flags, hash_dict)
from experiments.data_provider import DATASET_CONFIGS

import experiments.spot_regression_exp.run_regression_exp
import numpy as np
import datetime
import itertools
import argparse
import os

MODEL_SPECIFIC_CONFIG = {
    'OnlySim_tuned': {
    },
    'BNN_SVGD': {
        'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -1., 'max': 4.},
        'min_train_steps': {'values': [5000, 10000, 20000]},
        'max_train_steps': {'values': [5000, 10000, 20000]},
    },
    'BNN_FSVGD': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'bandwidth_gp_prior': {'distribution': 'log_uniform', 'min': -2., 'max': 0.},
        'min_train_steps': {'values': [5000, 10000, 20000]},
        'max_train_steps': {'values': [5000, 10000, 20000]},
        'num_measurement_points': {'values': [32]},
    },
    'BNN_FSVGD_SimPrior_gp': {
        'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
        'min_train_steps': {'values': [10000, 40000, 90000]},
        'min_train_steps': {'values': [10000, 40000, 90000]},
        'num_measurement_points': {'values': [32]},
        'num_f_samples': {'values': [1024]},
    },
    # 'BNN_FSVGD_SimPrior_ssge': {
    #     'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 1.0},
    #     'min_train_steps': {'values': [10000, 20000, 40000]},
    #     'max_train_steps': {'values': [10000, 20000, 40000]},
    #     'num_measurement_points': {'values': [8, 16, 32]},
    #     'num_f_samples': {'values': [512]},
    #     'bandwidth_score_estim': {'distribution': 'log_uniform_10', 'min': -0.5, 'max': 1.},
    # },
    # 'BNN_FSVGD_SimPrior_nu-method': {
    #     'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
    #     'min_train_steps': {'values': [30000, 40000, 50000]},
    #     'max_train_steps': {'values': [30000, 40000, 50000]},
    #     'num_measurement_points': {'values': [32]},
    #     'num_f_samples': {'values': [512]},
    #     #'bandwidth_score_estim': {'distribution': 'log_uniform_10', 'min': -0.5, 'max': 0.5},
    #     'bandwidth_score_estim': {'distribution': 'uniform', 'min': 1.0, 'max': 2.0},
    # },
    # 'BNN_FSVGD_SimPrior_kde': {
    #     'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 1.0},
    #     'min_train_steps': {'values': [60000, 80000, 100000]},
    #     'max_train_steps': {'values': [60000, 80000, 100000]},
    #     'num_measurement_points': {'values': [16, 32]},
    #     'num_f_samples': {'values': [1024, 2056]},
    # },
    # 'BNN_FSVGD_SimPrior_gp+nu-method': {
    #     'bandwidth_svgd': {'distribution': 'log_uniform_10', 'min': -1.0, 'max': 0.0},
    #     'min_train_steps': {'values': [80000]},
    #     'max_train_steps': {'values': [80000]},
    #     'num_measurement_points': {'values': [16, 32]},
    #     'num_f_samples': {'values': [512]},
    #     'switch_score_estimator_frac': {'values': [0.6667]},
    #     'bandwidth_score_estim': {'distribution': 'log_uniform_10', 'min': 0.0, 'max': 0.5},
    # },
    # 'BNN_SVGD_DistillPrior': {
    #     'bandwidth_svgd': {'distribution': 'log_uniform', 'min': -2., 'max': 2.},
    #     'min_train_steps': {'values': [20000, 40000]},
    #     'max_train_steps': {'values': [20000, 40000]},
    #     'num_measurement_points': {'values': [8, 16, 32]},
    #     'num_f_samples': {'values': [64, 128, 256]},
    #     'num_distill_steps': {'values': [30000, 60000]},
    # },
}


def main(args):
    # setup random seeds
    rds = np.random.RandomState(args.seed)
    model_seeds = list(rds.randint(0, 10**6, size=(100,)))
    data_seeds = list(rds.randint(0, 10**6, size=(100,)))

    sweep_config = {
        'data_source': {'value': args.data_source},
        'num_samples_train': DATASET_CONFIGS[args.data_source]['num_samples_train'],
        'model': {'value': args.model},
        'learn_likelihood_std': {'value': args.learn_likelihood_std},
        'pred_diff': {'value': args.pred_diff},
        'num_particles': {'value': 20},
        'data_batch_size': {'value': 8},
    }
    # update with model specific sweep ranges
    assert args.model in MODEL_SPECIFIC_CONFIG
    sweep_config.update(MODEL_SPECIFIC_CONFIG[args.model])

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.data_source}_{args.model}')

    command_list = []
    output_file_list = []
    for _ in range(args.num_hparam_samples):
        flags = sample_param_flags(sweep_config)

        exp_result_folder = os.path.join(exp_path, hash_dict(flags))
        flags['exp_result_folder'] = exp_result_folder

        for model_seed, data_seed in itertools.product(model_seeds[:args.num_model_seeds],
                                                       data_seeds[:args.num_data_seeds]):
            cmd = generate_base_command(experiments.spot_regression_exp.run_regression_exp,
                                        flags=dict(**flags, **{'model_seed': model_seed, 'data_seed': data_seed}))
            command_list.append(cmd)
            output_file_list.append(os.path.join(exp_result_folder, f'{model_seed}_{data_seed}.out'))

    generate_run_commands(command_list, output_file_list, num_cpus=args.num_cpus,
                          num_gpus=1 if args.gpu else 0, mode=args.run_mode, prompt=not args.yes)


if __name__ == '__main__':
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description='Meta-BO run')

    # sweep args
    parser.add_argument('--num_hparam_samples', type=int, default=5)
    parser.add_argument('--num_model_seeds', type=int, default=3, help='number of model seeds per hparam')
    parser.add_argument('--num_data_seeds', type=int, default=3, help='number of model seeds per hparam')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--run_mode', type=str, default='local')

    # general args
    parser.add_argument('--exp_name', type=str, default=f'test_{current_date}')
    parser.add_argument('--seed', type=int, default=94563)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--yes', default=False, action='store_true')

    # data parameters
    parser.add_argument('--data_source', type=str, default='spot_real')

    # standard BNN parameters
    parser.add_argument('--model', type=str, default='BNN_FSVGD_SimPrior_gp')
    parser.add_argument('--learn_likelihood_std', type=int, default=0)
    parser.add_argument('--pred_diff', type=int, default=1)

    args = parser.parse_args()
    main(args)
