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
    random_seed: int,
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
        random_seed=random_seed,
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
    model_key, data_key = jr.split(jr.PRNGKey(random_seed), 2)
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
        num_offline_collected_transitions=num_offline_collected_transitions,
        test_data_ratio=test_data_ratio,
        eval_sac_only_from_init_states=bool(eval_only_on_init_states),
        num_init_points_to_bs_for_sac_learning=num_init_points_to_bs_for_sac_learning,
        train_sac_only_from_init_states=bool(train_sac_only_from_init_states),
        wandb_logging=wandb_logging,
    )
    policy, params, metrics = rl_from_offline_data.prepare_policy_from_offline_data()

    # evaluate learned model
    rl_from_offline_data.eval_sim_model_on_dedicated_data()

    # evaluate policy on default simulator
    rl_from_offline_data.evaluate_policy_on_the_simulator(
        policy,
        key=key_evaluation_pretrained_bnn,
        num_evals=50,
        save_traj_dir=(
            f"/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/sim/{wandb.run.id}"
            if save_traj_local
            else None
        ),
    )

    # for testing: prepare policy as in real robot and then test

    # get offline trained agent
    def get_offline_trained_agent(
        state_dim: int,
        action_dim: int,
        goal_dim: int,
    ):
        import pickle
        import yaml

        # fetch learned policy
        wandb_api = wandb.Api()
        project_name = wandb.run.project
        run_id = run_id = wandb.run.id
        local_dir = "saved_data"

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        run = wandb_api.run(f"{project_name}/{run_id}")
        run.file("models/parameters.pkl").download(
            replace=True, root=os.path.join(local_dir)
        )

        # get reward config
        reward_keys = [
            "encode_angle",
            "ctrl_cost_weight",
            "margin_factor",
            "ctrl_diff_weight",
        ]
        reward_config = {}
        for key in reward_keys:
            reward_config[key] = run.config[key]

        # save reward config
        with open(os.path.join(local_dir, "reward_config.yaml"), "w") as file:
            yaml.dump(reward_config, file)

        # get policy params
        policy_params = pickle.load(
            open(os.path.join(local_dir, "models/parameters.pkl"), "rb")
        )

        # get reward config
        reward_config = yaml.load(
            open(os.path.join(local_dir, "reward_config.yaml"), "r"),
            Loader=yaml.Loader,
        )

        SAC_KWARGS_TEST = dict(
            num_timesteps=1_000_000,
            num_evals=20,
            reward_scaling=10,
            episode_length=50,
            episode_length_eval=2 * 50,
            action_repeat=1,
            discounting=0.99,
            lr_policy=3e-4,
            lr_alpha=3e-4,
            lr_q=3e-4,
            num_envs=64,
            batch_size=64,
            grad_updates_per_step=16 * 64,
            num_env_steps_between_updates=16,
            tau=0.005,
            wd_policy=0,
            wd_q=0,
            wd_alpha=0,
            num_eval_envs=2 * 64,
            max_replay_size=5 * 10**4,
            min_replay_size=2**11,
            policy_hidden_layer_sizes=(64, 64),
            critic_hidden_layer_sizes=(64, 64),
            normalize_observations=True,
            deterministic_eval=True,
            wandb_logging=False,
        )

        return policy_params, reward_config, SAC_KWARGS_TEST

    state_dim = 13
    action_dim = 6
    goal_dim = 3

    policy_params, reward_config, SAC_KWARGS_TEST = get_offline_trained_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    rl_from_offline_data_test = RLFromOfflineData(
        sac_kwargs=SAC_KWARGS_TEST,
        x_train=jax.numpy.zeros((10, state_dim + goal_dim + action_dim)),
        y_train=jax.numpy.zeros((10, state_dim)),
        x_test=jax.numpy.zeros((10, state_dim + goal_dim + action_dim)),
        y_test=jax.numpy.zeros((10, state_dim)),
        spot_reward_kwargs=reward_config,
    )
    policy_test = rl_from_offline_data_test.prepare_policy(params=policy_params)

    rl_from_offline_data_test.evaluate_policy_on_the_simulator_test(
        policy_test,
        key=key_evaluation_pretrained_bnn,
        num_evals=50,
        save_traj_dir=(
            f"/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/sim/{wandb.run.id}"
            if save_traj_local
            else None
        ),
    )

    if wandb_logging:
        wandb.finish()


def main(args):
    experiment(
        random_seed=args.random_seed,
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
        save_traj_local=args.save_traj_local,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # prevent TF from reserving all GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # parameters
    parser.add_argument("--random_seed", type=int, default=922852)
    parser.add_argument("--horizon_len", type=int, default=120)
    parser.add_argument("--sac_num_env_steps", type=int, default=10_000)
    parser.add_argument("--project_name", type=str, default="testing")
    parser.add_argument("--best_policy", type=int, default=1)
    parser.add_argument("--margin_factor", type=float, default=5.0)
    parser.add_argument("--ctrl_cost_weight", type=float, default=0.01)
    parser.add_argument("--ctrl_diff_weight", type=float, default=0.01)
    parser.add_argument("--num_offline_collected_transitions", type=int, default=5_000)
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
