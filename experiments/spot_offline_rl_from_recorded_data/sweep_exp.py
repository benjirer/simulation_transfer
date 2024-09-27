import os
import itertools
from experiments.util import generate_run_commands
from typing import List, Optional
import sys


def main(model: str, mode: str, num_cpus: int, num_gpus: int, mem: int):
    parameters = {
        # parameters
        # "model_seed": [922852, 123456, 654321],
        "model_seed": [922852],
        "data_seed": [0],
        "horizon_len": [150],
        "sac_num_env_steps": [2_000_000],
        "project_name": ["spot_offline_policy_v2"],
        "best_policy": [1],
        "margin_factor": [5.0],
        "ctrl_cost_weight": [0.005, 0.01, 0.02, 0.05],
        "ctrl_diff_weight": [0.01],
        "num_offline_collected_transitions": [2000],
        "test_data_ratio": [0.1],
        "share_of_x0s_in_sac_buffer": [0.5],
        "eval_only_on_init_states": [0],
        "eval_on_all_offline_data": [1],
        "train_sac_only_from_init_states": [0],
        "obtain_consecutive_data": [0],
        "wandb_logging": [True],
        "save_traj_local": [False],
    }

    parameters_bnn = {
        # model parameters
        "learnable_likelihood_std": ["yes"],
        "include_aleatoric_noise": [1],
        "best_bnn_model": [1],
        "predict_difference": [0],
        "use_sim_prior": [1],
        "use_grey_box": [0],
        "use_sim_model": [0],
        "num_measurement_points": [32],
        "bnn_batch_size": [32],
        "likelihood_exponent": [1.0],
        "bandwidth_svgd": [5.0],
        "num_epochs": [70],
        "max_train_steps": [150_000],
        "min_train_steps": [10_000],
        "length_scale_aditive_sim_gp": [1.0],
        "lr": [1e-3],
    }

    if model == "sim":
        script_path = (
            "/cluster/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_sim.py"
            if mode == "euler"
            else "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_sim.py"
        )
    elif model == "bnn":
        parameters.update(parameters_bnn)
        script_path = (
            "/cluster/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_bnn.py"
            if mode == "euler"
            else "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_bnn.py"
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
        prompt=True,
    )


if __name__ == "__main__":
    """Experiment settings"""
    model = "sim"
    mode = "local"
    num_cpus = 1
    num_gpus = 1
    mem = 16000

    main(model, mode, num_cpus, num_gpus, mem)
