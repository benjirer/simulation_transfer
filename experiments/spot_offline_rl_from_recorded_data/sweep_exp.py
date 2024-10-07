import os
import itertools
from experiments.util import generate_run_commands
from typing import List, Optional
import sys
import random

def main(model: str, mode: str, num_cpus: int, num_gpus: int, mem: int):

    random.seed(0)
    random_seed = random.sample(range(1, 1_000_000), 3)
    random_seed = [0]
    num_offline_collected_transitions = [1000, 2500, 5000]
    num_offline_collected_transitions = [5000]

    parameters = {
        # parameters general
        "random_seed": random_seed,
        "num_offline_collected_transitions": num_offline_collected_transitions,
        "test_data_ratio": [0.1],
        "wandb_logging": [True],
        "project_name": ["jitter_testing"],
    }

    parameters_sac = {
        # parameters sac
        "horizon_len": [150],
        "sac_num_env_steps": [2_000_000],
        "best_policy": [1],
        "margin_factor": [5.0],
        "ctrl_cost_weight": [0.005],
        "ctrl_diff_weight": [0.01],
        "share_of_x0s_in_sac_buffer": [0.5],
        "eval_only_on_init_states": [0],
        "eval_on_all_offline_data": [1],
        "train_sac_only_from_init_states": [0],
        "obtain_consecutive_data": [0],
        "save_traj_local": [True],
    }

    parameters_bnn = {
        # model parameters
        "learnable_likelihood_std": ["yes"],
        "include_aleatoric_noise": [1],
        "best_bnn_model": [1],
        "predict_difference": [0],
        "use_sim_prior": [1] if model == "bnn-sim-fsvgd" else [0],
        "use_grey_box": [0],
        "use_sim_model": [0],
        "num_measurement_points": [32],
        "bnn_batch_size": [32],
        "likelihood_exponent": [1.0],
        "bandwidth_svgd": [5.0],
        "num_epochs": [3],
        "max_train_steps": [150_000],
        "min_train_steps": [10_000],
        "length_scale_aditive_sim_gp": [1.0],
        "lr": [1e-3],
    }

    if model == "sim":
        parameters.update(parameters_sac)
        script_path = (
            "/cluster/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_sim.py"
            if mode == "euler"
            else "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_sim.py"
        )
    elif model == "bnn-fsvgd" or model == "bnn-sim-fsvgd":
        parameters.update(parameters_bnn)
        script_path = (
            "/cluster/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_bnn.py"
            if mode == "euler"
            else "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_bnn.py"
        )
    else:
        raise ValueError(f"Model {model} not implemented")

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
    # models = ["sim", "bnn-fsvgd", "bnn-sim-fsvgd"]
    models = ["bnn-sim-fsvgd"]
    mode = "local"
    num_cpus = 1
    num_gpus = 1
    mem = 16000

    for model in models:
        main(model, mode, num_cpus, num_gpus, mem)
