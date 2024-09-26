import os
import itertools

# parameters
parameters = {
    # parameters
    "model_seed": [922852, 123456, 654321],
    "data_seed": [0],
    "horizon_len": [120],
    "sac_num_env_steps": [2_000_000],
    "project_name": ["spot_offline_policy"],
    "best_policy": [1],
    "margin_factor": [1.0, 5.0, 10.0],
    "ctrl_cost_weight": [0.01],
    "ctrl_diff_weight": [0.01],
    "num_offline_collected_transitions": [4100, 8200],
    "test_data_ratio": [0.1],
    "share_of_x0s_in_sac_buffer": [0.5],
    "eval_only_on_init_states": [0],
    "eval_on_all_offline_data": [1],
    "train_sac_only_from_init_states": [0],
    "obtain_consecutive_data": [0],
    "wandb_logging": [True],

    # model parameters
    "learnable_likelihood_std": ["yes", "no"],
    "include_aleatoric_noise": [0, 1],
    "best_bnn_model": [1],
    "predict_difference": [0],
    "use_sim_prior": [1, 0],
    "use_grey_box": [0],
    "use_sim_model": [0],
    "num_measurement_points": [32],
    "bnn_batch_size": [32],
    "likelihood_exponent": [1.0],
    "bandwidth_svgd": [5.0],
    "num_epochs": [10],
    "max_train_steps": [100_000],
    "min_train_steps": [10_000],
    "length_scale_aditive_sim_gp": [1.0],
    "lr": [1e-3],
}

python_path = "/home/bhoffman/anaconda3/envs/sim_transfer/bin/python"
script_path = "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_bnn.py"

# generate all combinations
keys = list(parameters.keys())
values = list(parameters.values())
all_combinations = list(itertools.product(*values))

# confirmation
total_experiments = len(all_combinations)
print(f"You are about to run {total_experiments} experiments. Do you want to continue? (y/n): ", end='')
response = input().strip().lower()
if response != 'y':
    print("Aborting experiments.")
    exit()

# run experiments
for combination in all_combinations:
    cmd = f"{python_path} {script_path} "
    for key, value in zip(keys, combination):
        cmd += f"--{key} {value} "
    print(f"Running command: {cmd}")
    os.system(cmd)
