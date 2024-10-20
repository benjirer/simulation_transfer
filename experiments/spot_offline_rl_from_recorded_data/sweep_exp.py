import os
import itertools
from experiments.util import generate_run_commands
from typing import List, Optional
import sys
import random

def main(model: str, mode: str, num_cpus: int, num_gpus: int, mem: int):

    random.seed(0)
    # random_seed = random.sample(range(1, 1_000_000), 3)
    random_seed = [42, 9126, 1913244]
    # random_seed = [42]
    num_offline_collected_transitions = [800, 2000, 5000]
    # num_offline_collected_transitions = [5000]

    parameters = {
        # parameters general
        "num_frame_stack": [2],
        "random_seed": random_seed,
        "num_offline_collected_transitions": num_offline_collected_transitions,
        "test_data_ratio": [0.15],
        "wandb_logging": [True],
        "project_name": ["policy_testing_full_v9"],
        "obtain_consecutive_data": [1],
        "save_traj_local": [False],
    }

    parameters_rl = {
        # parameters rl
        "horizon_len": [120],
        "sac_num_env_steps": [2_500_000],
        "best_policy": [1],
        "margin_factor": [10.0],
        "ctrl_cost_weight": [0.05],
        "ctrl_diff_weight": [0.3, 0.4],
        "share_of_x0s_in_sac_buffer": [0.5],
        "eval_only_on_init_states": [1],
        "eval_on_all_offline_data": [1],
        "train_sac_only_from_init_states": [0],
    }

    parameters_model = {
        # model parameters
        "learnable_likelihood_std": ["yes"],
        "include_aleatoric_noise": [1],
        "best_bnn_model": [1],
        "predict_difference": [1],
        "use_sim_prior": [1] if model == "bnn-sim-fsvgd" else [0],
        "use_sim_model": [1] if model == "sim-model" else [0],
        "num_measurement_points": [48],
        "bnn_batch_size": [48],
        "likelihood_exponent": [1.0],
        "bandwidth_svgd": [5.0],
        "num_epochs": [70],
        "max_train_steps": [150_000],
        "min_train_steps": [10_000],
        "num_sim_fitting_steps": [40_000],
        "length_scale_aditive_sim_gp": [1.0],
        "lr": [1e-3],
    }

    parameters.update(parameters_rl)
    parameters.update(parameters_model)
    script_path = (
        "/cluster/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp.py"
        if mode == "euler"
        else "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp.py"
    )

    python_path = sys.executable

    # generate all combinations
    keys = list(parameters.keys())
    values = list(parameters.values())
    all_combinations = list(itertools.product(*values))

    # generate commands
    command_list = []
    for combination in all_combinations:
        cmd = f"{python_path} {script_path} "
        for key, value in zip(keys, combination):
            cmd += f"--{key} {value} "
        command_list.append(cmd)

    # run experiments
    generate_run_commands(
        command_list=command_list,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        dry=False,
        mem=mem,
        duration="0:29:00",
        mode=mode,
        prompt=False,
    )


if __name__ == "__main__":
    """Experiment settings"""
    models = ["sim-model", "bnn-sim-fsvgd", "bnn-fsvgd"]
    # models = ["bnn-sim-fsvgd"]
    mode = "local"
    num_cpus = 1
    num_gpus = 1
    mem = 16000

    for model in models:
        main(model, mode, num_cpus, num_gpus, mem)
