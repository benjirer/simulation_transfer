import argparse

import jax.nn
import jax.random as jr
import wandb
import os

from experiments.data_provider import provide_data_and_sim, _SPOT_NOISE_STD_ENCODED
from sim_transfer.rl.spot_sim_rl_on_offline_data import RLFromOfflineData


def experiment(
    # variable parameters
    horizon_len: int,
    model_seed: int,
    data_seed: int,
    project_name: str,
    sac_num_env_steps: int,
    best_policy: int,
    margin_factor: float,
    ctrl_cost_weight: float,
    ctrl_diff_weight: float,
    num_offline_collected_transitions: int,
    share_of_x0s_in_sac_buffer: float,
    eval_only_on_init_states: int,
    train_sac_only_from_init_states: int,
    
    # default parameters
    eval_on_all_offline_data: int = 1,
    test_data_ratio: float = 0.1,
    default_num_init_points_to_bs_for_sac_learning=1000,
    obtain_consecutive_data: int = 1,
    wandb_logging: bool = True,
    save_traj_local: bool = True,
):
    # set parameters
    config_dict = dict(
        num_offline_data=num_offline_collected_transitions,
        share_of_x0s=share_of_x0s_in_sac_buffer,
        sac_only_from_is=train_sac_only_from_init_states,
    )

    group_name = "_".join(
        list(
            str(key) + "=" + str(value)
            for key, value in config_dict.items()
            if key != "seed"
        )
    )

    spot_reward_kwargs = dict(
        encode_angle=True,
        ctrl_cost_weight=ctrl_cost_weight,
        margin_factor=margin_factor,
        ctrl_diff_weight=ctrl_diff_weight,
    )

    NUM_ENV_STEPS_BETWEEN_UPDATES = 16
    NUM_ENVS = 64

    SAC_KWARGS = dict(
        num_timesteps=sac_num_env_steps,
        num_evals=20,
        reward_scaling=1,
        episode_length=horizon_len,
        episode_length_eval=horizon_len,
        action_repeat=1,
        discounting=0.99,
        lr_policy=1e-4,
        lr_alpha=1e-4,
        lr_q=1e-4,
        num_envs=NUM_ENVS,
        batch_size=64,
        grad_updates_per_step=NUM_ENV_STEPS_BETWEEN_UPDATES * NUM_ENVS,
        num_env_steps_between_updates=NUM_ENV_STEPS_BETWEEN_UPDATES,
        tau=0.005,
        wd_policy=0,
        wd_q=0,
        wd_alpha=0,
        num_eval_envs=2 * NUM_ENVS,
        max_replay_size=5 * 10**4,
        min_replay_size=2**11,
        policy_hidden_layer_sizes=(64, 64),
        critic_hidden_layer_sizes=(64, 64),
        normalize_observations=True,
        deterministic_eval=True,
        wandb_logging=wandb_logging,
        max_grad_norm=100,
    )

    config_dict = dict(
        horizon_len=horizon_len,
        model_seed=model_seed,
        data_seed=data_seed,
        sac_num_env_steps=sac_num_env_steps,
        best_policy=best_policy,
        margin_factor=margin_factor,
        ctrl_diff_weight=ctrl_diff_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        num_offline_collected_transitions=num_offline_collected_transitions,
        share_of_x0s_in_sac_buffer=share_of_x0s_in_sac_buffer,
        eval_only_on_init_states=eval_only_on_init_states,
        eval_on_all_offline_data=eval_on_all_offline_data,
        train_sac_only_from_init_states=train_sac_only_from_init_states,
    )

    total_config = SAC_KWARGS | config_dict | spot_reward_kwargs
    group = group_name
    if wandb_logging:
        wandb.init(
            project=project_name,
            group=group,
            config=total_config,
        )

    # deal with randomness
    model_key = jr.PRNGKey(model_seed)
    data_key = jr.PRNGKey(data_seed)
    int_data_seed = jr.randint(data_key, (), minval=0, maxval=2**13 - 1)
    key_offline_rl, key_evaluation_pretrained_bnn = jr.split(model_key, 2)

    # get data and sim
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source="spot_real_with_goal",
        data_spec={
            "num_samples_train": int(num_offline_collected_transitions),
            "sampling": "consecutive" if obtain_consecutive_data else "iid",
        },
        data_seed=int(int_data_seed),
    )

    print(
        "Original data from provider: x_train.shape",
        x_train.shape,
        "y_train.shape",
        y_train.shape,
        "x_test.shape",
        x_test.shape,
        "y_test.shape",
        y_test.shape,
    )

    num_init_points_to_bs_for_sac_learning = int(
        num_offline_collected_transitions
        * share_of_x0s_in_sac_buffer
        / (1 - share_of_x0s_in_sac_buffer)
    )
    if train_sac_only_from_init_states:
        num_init_points_to_bs_for_sac_learning = (
            default_num_init_points_to_bs_for_sac_learning
        )

    # perform experiment
    rl_from_offline_data = RLFromOfflineData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        key=key_offline_rl,
        sac_kwargs=SAC_KWARGS,
        spot_reward_kwargs=spot_reward_kwargs,
        return_best_policy=bool(best_policy),
        test_data_ratio=test_data_ratio,
        eval_sac_only_from_init_states=bool(eval_only_on_init_states),
        num_init_points_to_bs_for_sac_learning=num_init_points_to_bs_for_sac_learning,
        train_sac_only_from_init_states=bool(train_sac_only_from_init_states),
        wandb_logging=wandb_logging,
    )
    policy, params, metrics = rl_from_offline_data.prepare_policy_from_offline_data()

    # evaluate policy on simulator
    rl_from_offline_data.evaluate_policy_on_the_simulator(
        policy,
        key=key_evaluation_pretrained_bnn,
        num_evals=50,
        save_traj_dir=f"/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/sim/{wandb.run.id}" if save_traj_local else None,
    )

    if wandb_logging:
        wandb.finish()


def main(args):
    experiment(
        model_seed=args.model_seed,
        data_seed=args.data_seed,
        project_name=args.project_name,
        horizon_len=args.horizon_len,
        sac_num_env_steps=args.sac_num_env_steps,
        best_policy=args.best_policy,
        margin_factor=args.margin_factor,
        ctrl_cost_weight=args.ctrl_cost_weight,
        ctrl_diff_weight=args.ctrl_diff_weight,
        num_offline_collected_transitions=args.num_offline_collected_transitions,
        test_data_ratio=args.test_data_ratio,
        share_of_x0s_in_sac_buffer=args.share_of_x0s_in_sac_buffer,
        eval_only_on_init_states=args.eval_only_on_init_states,
        eval_on_all_offline_data=args.eval_on_all_offline_data,
        train_sac_only_from_init_states=args.train_sac_only_from_init_states,
        obtain_consecutive_data=args.obtain_consecutive_data,
        wandb_logging=args.wandb_logging,
        save_traj_local=args.save_traj_local
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # prevent TF from reserving all GPU memory
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # parameters
    parser.add_argument("--model_seed", type=int, default=922852)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--horizon_len", type=int, default=120)
    parser.add_argument("--sac_num_env_steps", type=int, default=2_000_000)
    parser.add_argument("--project_name", type=str, default="spot_offline_policy")
    parser.add_argument("--best_policy", type=int, default=1)
    parser.add_argument("--margin_factor", type=float, default=5.0)
    parser.add_argument("--ctrl_cost_weight", type=float, default=0.01)
    parser.add_argument("--ctrl_diff_weight", type=float, default=0.01)
    parser.add_argument("--num_offline_collected_transitions", type=int, default=4_100)
    parser.add_argument("--test_data_ratio", type=float, default=0.1)
    parser.add_argument("--share_of_x0s_in_sac_buffer", type=float, default=0.5)
    parser.add_argument("--eval_only_on_init_states", type=int, default=0)
    parser.add_argument("--eval_on_all_offline_data", type=int, default=1)
    parser.add_argument("--train_sac_only_from_init_states", type=int, default=0)
    parser.add_argument("--obtain_consecutive_data", type=int, default=0)
    parser.add_argument("--wandb_logging", type=bool, default=True)
    parser.add_argument("--save_traj_local", type=bool, default=True)
    args = parser.parse_args()
    main(args)
