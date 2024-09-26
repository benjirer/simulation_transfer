import os
import itertools

# parameters
parameters = {
    "model_seed": [922852, 123456],
    "data_seed": [0],
    "horizon_len": [50, 100],
    "sac_num_env_steps": [2_000_000, 3_000_000],
    "project_name": ["spot_offline_policy"],
    "best_policy": [1],
    "margin_factor": [5.0, 10.0],
    "ctrl_cost_weight": [0.005, 0.01],
    "ctrl_diff_weight": [0.01],
    "num_offline_collected_transitions": [2000, 4100],
    "test_data_ratio": [0.0],
    "share_of_x0s_in_sac_buffer": [0.5, 0.75],
    "eval_only_on_init_states": [0],
    "eval_on_all_offline_data": [1],
    "train_sac_only_from_init_states": [0, 1],
    "obtain_consecutive_data": [0],
    "wandb_logging": [True],
}

python_path = "/home/bhoffman/anaconda3/envs/sim_transfer/bin/python"
script_path = "/home/bhoffman/Documents/MT_FS24/simulation_transfer/experiments/spot_offline_rl_from_recorded_data/exp_sim.py"

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
