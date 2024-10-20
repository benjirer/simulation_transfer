import argparse

import jax.nn
import jax.random as jr
import wandb
import os

from experiments.data_provider import provide_data_and_sim, _SPOT_NOISE_STD_ENCODED
from sim_transfer.rl.spot_rl_on_offline_data import RLFromOfflineData

# imports for model
from sim_transfer.models import BNN_FSVGD_SimPrior, BNN_FSVGD, BNNGreyBox
from sim_transfer.sims.simulators import (
    AdditiveSim,
    PredictStateChangeWrapper,
    GaussianProcessSim,
)


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
    # variable model parameters
    learnable_likelihood_std: str,
    include_aleatoric_noise: int,
    best_bnn_model: int,
    predict_difference: int,
    use_sim_prior: int,
    use_sim_model: int,
    num_measurement_points: int,
    bnn_batch_size: int,
    # default parameters
    num_frame_stack: int = 0,
    eval_on_all_offline_data: int = 1,
    test_data_ratio: float = 0.1,
    default_num_init_points_to_bs_for_sac_learning=1000,
    obtain_consecutive_data: int = 1,
    wandb_logging: bool = True,
    save_traj_local: bool = True,
    # default model parameters
    likelihood_exponent: float = 1.0,
    bandwidth_svgd: float = 2.0,
    num_epochs: int = 50,
    max_train_steps: int = 100_000,
    min_train_steps: int = 40_000,
    num_sim_fitting_steps: int = 40_000,
    length_scale_aditive_sim_gp: float = 1.0,
    lr: float = 3e-4,
):
    # can only use one model at a time
    assert not (use_sim_prior and use_sim_model), "Can only use one model at a time"

    # set parameters
    bnn_train_steps = min(
        num_epochs * num_offline_collected_transitions, max_train_steps
    )
    bnn_train_steps = max(bnn_train_steps, min_train_steps)

    config_dict = dict(
        use_sim_prior=use_sim_prior,
        use_sim_model=use_sim_model,
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
        # parameters general
        num_frame_stack=num_frame_stack,
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
        obtain_consecutive_data=obtain_consecutive_data,
        include_aleatoric_noise=include_aleatoric_noise,
        # parameters model
        bnn_train_steps=bnn_train_steps,
        ll_std=learnable_likelihood_std,
        best_bnn_model=best_bnn_model,
        predict_difference=predict_difference,
        use_sim_prior=use_sim_prior,
        use_sim_model=use_sim_model,
        bnn_batch_size=bnn_batch_size,
        num_measurement_points=num_measurement_points,
        test_data_ratio=test_data_ratio,
        bandwidth_svgd=bandwidth_svgd,
        num_epochs=num_epochs,
        max_train_steps=max_train_steps,
        min_train_steps=min_train_steps,
        num_sim_fitting_steps=num_sim_fitting_steps,
        length_scale_aditive_sim_gp=length_scale_aditive_sim_gp,
        likelihood_exponent=likelihood_exponent,
    )

    total_config = SAC_KWARGS | config_dict | spot_reward_kwargs
    group = group_name + "_" + str(likelihood_exponent)
    if wandb_logging:
        wandb.init(
            project=project_name,
            group=group,
            config=total_config,
        )

    # deal with randomness
    data_key, model_key = jr.split(jr.PRNGKey(random_seed), 2)
    int_data_seed = jr.randint(data_key, (), minval=0, maxval=2**13 - 1)
    (
        key_bnn,
        key_offline_rl,
        key_evaluation_trained_bnn,
        key_evaluation_pretrained_bnn,
    ) = jr.split(model_key, 4)

    # get data and sim
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source="spot_real_with_goal",
        data_spec={
            "num_samples_train": int(num_offline_collected_transitions),
            "sampling": "consecutive" if obtain_consecutive_data else "iid",
            "num_stacked_actions": num_frame_stack,
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

    # set up model
    standard_bnn_params = {
        "input_size": sim.input_size,
        "output_size": sim.output_size,
        "rng_key": key_bnn,
        "likelihood_std": _SPOT_NOISE_STD_ENCODED,
        "normalize_data": True,
        "normalize_likelihood_std": True,
        "learn_likelihood_std": bool(learnable_likelihood_std),
        "likelihood_exponent": likelihood_exponent,
        "hidden_layer_sizes": [64, 64, 64],
        "data_batch_size": bnn_batch_size,
        "hidden_activation": jax.nn.leaky_relu,
    }
    

    # SIM-MODEL
    if use_sim_model:
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)
        base_bnn = BNN_FSVGD(
            **standard_bnn_params,
            normalization_stats=sim.normalization_stats,
            num_train_steps=bnn_train_steps,
            domain=sim.domain,
            lr=lr,
            bandwidth_svgd=bandwidth_svgd,
        )
        model = BNNGreyBox(
            base_bnn=base_bnn,
            sim=sim,
            use_base_bnn=False,
            num_sim_model_train_steps=num_sim_fitting_steps,
        )
    # SIM-FSVGD
    elif use_sim_prior:
        OUPUTSCALE_SPOT = [
            0.2,  # base_x
            0.2,  # base_y
            0.02,  # base_theta_sin
            0.02,  # base_theta_cos
            0.02,  # base_vel_x
            0.02,  # base_vel_y
            0.02,  # base_ang_vel
            0.2,  # ee_x
            0.2,  # ee_y
            0.02,  # ee_z
            0.02,  # ee_vx
            0.02,  # ee_vy
            0.02,  # ee_vz
        ]
        # OUPUTSCALE_SPOT = 1.0
        sim = AdditiveSim(
            base_sims=[
                sim,
                GaussianProcessSim(
                    sim.input_size,
                    sim.output_size,
                    output_scale=OUPUTSCALE_SPOT,
                    length_scale=length_scale_aditive_sim_gp,
                    consider_only_first_k_dims=None,
                ),
            ]
        )
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)

        model = BNN_FSVGD_SimPrior(
            **standard_bnn_params,
            normalization_stats=sim.normalization_stats,
            domain=sim.domain,
            function_sim=sim,
            score_estimator="gp",
            num_train_steps=bnn_train_steps,
            num_f_samples=256,
            lr=lr,
            bandwidth_svgd=bandwidth_svgd,
            num_measurement_points=num_measurement_points,
        )
    # BNN-FSVGD
    else:
        if predict_difference:
            sim = PredictStateChangeWrapper(sim)

        model = BNN_FSVGD(
            **standard_bnn_params,
            normalization_stats=sim.normalization_stats,
            num_train_steps=bnn_train_steps,
            domain=sim.domain,
            lr=lr,
            bandwidth_svgd=bandwidth_svgd,
        )

    # set up RL experiment
    num_init_points_to_bs_for_sac_learning = int(
        num_offline_collected_transitions
        * share_of_x0s_in_sac_buffer
        / (1 - share_of_x0s_in_sac_buffer)
    )
    if train_sac_only_from_init_states:
        num_init_points_to_bs_for_sac_learning = (
            default_num_init_points_to_bs_for_sac_learning
        )

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
        num_sim_fitting_steps=num_sim_fitting_steps,
        test_data_ratio=test_data_ratio,
        num_frame_stack=num_frame_stack,
        eval_sac_only_from_init_states=bool(eval_only_on_init_states),
        num_init_points_to_bs_for_sac_learning=num_init_points_to_bs_for_sac_learning,
        train_sac_only_from_init_states=bool(train_sac_only_from_init_states),
        wandb_logging=wandb_logging,
        bnn_model=model,
        include_aleatoric_noise=bool(include_aleatoric_noise),
        predict_difference=bool(predict_difference),
        eval_bnn_model_on_all_offline_data=bool(eval_on_all_offline_data),
    )

    # get policy from offline data
    policy, params, metrics, bnn_model = (
        rl_from_offline_data.prepare_policy_from_offline_data(
            bnn_train_steps=bnn_train_steps, return_best_bnn=bool(best_bnn_model)
        )
    )

    skip_eval = False
    if not skip_eval:
        # evaluate learned model
        # rl_from_offline_data.eval_model_on_dedicated_data(bnn_model=bnn_model)

        # evaluate policy on learned model
        rl_from_offline_data.evaluate_policy(
            policy,
            bnn_model,
            key=key_evaluation_trained_bnn,
            num_evals=10,
            save_traj_dir=(
                f"/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/bnn/{wandb.run.id}"
                if save_traj_local
                else None
            ),
        )

        # evaluate policy on default simulator
        rl_from_offline_data.evaluate_policy_on_the_simulator(
            policy,
            key=key_evaluation_pretrained_bnn,
            num_evals=10,
            save_traj_dir=(
                f"/home/bhoffman/Documents/MT_FS24/simulation_transfer/results/policies_traj/bnn/{wandb.run.id}"
                if save_traj_local
                else None
            ),
        )

    if wandb_logging:
        wandb.finish()


def main(args):
    experiment(
        # parameters general
        num_frame_stack=args.num_frame_stack,
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
        # model parameters
        learnable_likelihood_std=args.learnable_likelihood_std,
        include_aleatoric_noise=args.include_aleatoric_noise,
        best_bnn_model=args.best_bnn_model,
        predict_difference=args.predict_difference,
        use_sim_prior=args.use_sim_prior,
        use_sim_model=args.use_sim_model,
        num_measurement_points=args.num_measurement_points,
        bnn_batch_size=args.bnn_batch_size,
        likelihood_exponent=args.likelihood_exponent,
        bandwidth_svgd=args.bandwidth_svgd,
        num_epochs=args.num_epochs,
        max_train_steps=args.max_train_steps,
        min_train_steps=args.min_train_steps,
        num_sim_fitting_steps=args.num_sim_fitting_steps,
        length_scale_aditive_sim_gp=args.length_scale_aditive_sim_gp,
        lr=args.lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # prevent TF from reserving all GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # parameters general
    parser.add_argument("--num_frame_stack", type=int, default=0)
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

    # model parameters
    parser.add_argument("--learnable_likelihood_std", type=str, default="yes")
    parser.add_argument("--include_aleatoric_noise", type=int, default=1)
    parser.add_argument("--best_bnn_model", type=int, default=1)
    parser.add_argument("--predict_difference", type=int, default=0)
    parser.add_argument("--use_sim_prior", type=int, default=1)
    parser.add_argument("--use_sim_model", type=int, default=0)
    parser.add_argument("--num_measurement_points", type=int, default=32)
    parser.add_argument("--bnn_batch_size", type=int, default=32)
    parser.add_argument("--likelihood_exponent", type=float, default=1.0)
    parser.add_argument("--bandwidth_svgd", type=float, default=5.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=100_000)
    parser.add_argument("--min_train_steps", type=int, default=10_000)
    parser.add_argument("--num_sim_fitting_steps", type=int, default=40_000)
    parser.add_argument("--length_scale_aditive_sim_gp", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
