import pickle
import pandas as pd
import os
import jax.numpy as jnp
import jax
import optax
import numpy as np
import argparse
import glob
from functools import partial
from matplotlib import pyplot as plt
from typing import Tuple, List, Union, Dict
import wandb
from pprint import pprint

from experiments.util import get_trajectory_windows
from experiments.data_provider import _load_spot_datasets
from sim_transfer.sims.dynamics_models import SpotDynamicsModel, SpotParams
from sim_transfer.sims.simulators import SpotSim
from sim_transfer.sims.util import angle_diff
from sim_transfer.sims.util import encode_angles as encode_angles_fn
from sim_transfer.sims.util import decode_angles as decode_angles_fn
from experiments.data_provider import delay_and_stack_spot_actions


from brax.training.types import Transition
import tensorflow_probability.substrates.jax.distributions as tfd


def extra_eval_learned_model(
    spot_learned_params,
    num_offline_collected_transitions,
    wandb_logging,
    use_all_data=False,
):
    """Extra evaluation of the learned model on real data."""

    # load measured data for testing and eval
    if use_all_data:
        dir_path = (
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_v0",
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_v1",
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_v2",
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_v3",
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/recordings_spot_v4",
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/test_data_spot",
        )
        eval_trajectories_paths = sorted(
            [
                os.path.join(dir_path, f)
                for d in dir_path
                for f in os.listdir(d)
                if f.endswith(".pickle")
            ]
        )
    else:
        dir_path = (
            "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/test_data_spot"
        )
        eval_trajectories_paths = sorted(
            [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(".pickle")
            ]
        )
    eval_trajectories, eval_trajectories_id = _load_spot_datasets(
        eval_trajectories_paths
    ), [os.path.basename(f).split(".")[0] for f in eval_trajectories_paths]

    # extra evaluation settings
    action_delay_base = 0
    action_delay_ee = 0
    step_range = min(200, min([traj[0].shape[0] for traj in eval_trajectories]))
    state_labels = [
        "base_x",
        "base_y",
        "base_theta",
        "base_vel_x",
        "base_vel_y",
        "base_ang_vel",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_vx",
        "ee_vy",
        "ee_vz",
    ]

    # setup extra eval metrics (rmse over all trajectories)
    extra_eval_metrics = {}

    # iterate over eval trajectories
    for traj, traj_id in zip(eval_trajectories, eval_trajectories_id):
        testing_x_pre_org, testing_u_pre_org, testing_y_org = traj

        # cut data
        testing_x_pre = testing_x_pre_org[:step_range]
        testing_u_pre = testing_u_pre_org[:step_range]
        testing_y = testing_y_org[:step_range]

        # apply action delay
        testing_u_pre = delay_and_stack_spot_actions(
            u=testing_u_pre,
            action_stacking=False,
            action_delay_base=action_delay_base,
            action_delay_ee=action_delay_ee,
        )

        # prepare data
        testing_x_pre = encode_angles_fn(testing_x_pre, 2)
        testing_x = jnp.concatenate([testing_x_pre, testing_u_pre], axis=1)

        # prepare error metrics
        model_errors_max_ee = {}
        model_errors_total_ee = {}
        model_errors_max_base = {}
        model_errors_total_base = {}
        model_errors_max_base_theta = {}
        model_errors_total_base_theta = {}

        # simulate trajectory with learned model
        model = SpotSim(encode_angle=True, spot_model_params=spot_learned_params)
        model_name = f"sim_{num_offline_collected_transitions}"
        y_pred_testing = []
        x_state = testing_x[0:1]
        for i in range(testing_x.shape[0]):
            y_pred = model.evaluate_sim(x_state, SpotParams(**spot_learned_params))
            y_pred_testing.append(y_pred[0])
            if i < testing_x.shape[0] - 1:
                u_next = testing_u_pre[i + 1 : i + 2]
                x_state = jnp.concatenate([y_pred, u_next], axis=1)
        y_pred_testing = jnp.array(y_pred_testing)

        y_pred_testing = decode_angles_fn(y_pred_testing, 2)

        # calculate errors for plotting
        ee_pos_error_running = jnp.linalg.norm(
            y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1
        )
        ee_pos_error_cumulative = jnp.cumsum(ee_pos_error_running)
        ee_pos_error_max = jnp.max(ee_pos_error_running)
        ee_pos_error_total = jnp.sum(ee_pos_error_running)
        model_errors_max_ee[model_name] = ee_pos_error_max
        model_errors_total_ee[model_name] = ee_pos_error_total

        base_pos_error_running = jnp.linalg.norm(
            y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1
        )
        base_pos_error_cumulative = jnp.cumsum(base_pos_error_running)
        base_pos_error_max = jnp.max(base_pos_error_running)
        base_pos_error_total = jnp.sum(base_pos_error_running)
        model_errors_max_base[model_name] = base_pos_error_max
        model_errors_total_base[model_name] = base_pos_error_total

        base_theta_error_running = jnp.linalg.norm(
            y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1
        )
        base_theta_error_cumulative = jnp.cumsum(base_theta_error_running)
        base_theta_error_max = jnp.max(base_theta_error_running)
        base_theta_error_total = jnp.sum(base_theta_error_running)
        model_errors_max_base_theta[model_name] = base_theta_error_max
        model_errors_total_base_theta[model_name] = base_theta_error_total

        # fill extra evals metrics for current trajectory
        for state_label in state_labels:
            state_idx = state_labels.index(state_label)
            state_error = (y_pred_testing[:, state_idx] - testing_y[:, state_idx]) ** 2
            extra_eval_metrics[f"{state_label}_rmse"] = (
                extra_eval_metrics.get(
                    f"{state_label}_rmse", jnp.zeros(state_error.shape)
                )
                + state_error
            )
        base_pos_error = (
            jnp.linalg.norm(y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1) ** 2
        )
        theta_error = (
            jnp.linalg.norm(y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1) ** 2
        )
        ee_pos_error = (
            jnp.linalg.norm(y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1) ** 2
        )
        extra_eval_metrics["base_pos_rmse"] = (
            extra_eval_metrics.get("base_pos_rmse", jnp.zeros(base_pos_error.shape))
            + base_pos_error
        )
        extra_eval_metrics["theta_rmse"] = (
            extra_eval_metrics.get(
                "theta_rmse", jnp.zeros(base_theta_error_running.shape)
            )
            + theta_error
        )
        extra_eval_metrics["ee_pos_rmse"] = (
            extra_eval_metrics.get("ee_pos_rmse", jnp.zeros(ee_pos_error.shape))
            + ee_pos_error
        )

        # detailed plot of trajectory rollout and errors
        # prepare plots
        fig, axs = plt.subplots(6, 2, figsize=(30, 15))
        fig_ee_error, axs_ee_error = plt.subplots(4, 1, figsize=(30, 15))
        fig_base_error, axs_base_error = plt.subplots(4, 2, figsize=(30, 15))

        # plot true data
        for i in range(6):
            axs[i, 0].plot(testing_y[:, i], label="true")
            axs[i, 0].set_title(state_labels[i])

            if i + 6 < testing_y.shape[-1]:
                axs[i, 1].plot(testing_y[:, i + 6], label="true")
                axs[i, 1].set_title(state_labels[i + 6])

        # prepare error plots
        axs_ee_error[0].set_title("Running ee position error")
        axs_ee_error[1].set_title("Running cumulative ee position error")
        axs_ee_error[2].set_title("Max ee position error")
        axs_ee_error[3].set_title("Total cumulative ee position error")
        axs_base_error[0, 0].set_title("Running base position error")
        axs_base_error[1, 0].set_title("Running cumulative base position error")
        axs_base_error[2, 0].set_title("Max base position error")
        axs_base_error[3, 0].set_title("Total cumulative base position error")
        axs_base_error[0, 1].set_title("Running base theta error")
        axs_base_error[1, 1].set_title("Running cumulative base theta error")
        axs_base_error[2, 1].set_title("Max base theta error")
        axs_base_error[3, 1].set_title("Total cumulative base theta error")

        # plot simulated trajectory
        for i in range(6):
            axs[i, 0].plot(
                y_pred_testing[:, i], label=f"pred {model_name}", linestyle="--"
            )

            if i + 6 < testing_y.shape[-1]:
                axs[i, 1].plot(
                    y_pred_testing[:, i + 6],
                    label=f"pred {model_name}",
                    linestyle="--",
                )

        # plot running and cumulative errors
        axs_ee_error[0].plot(ee_pos_error_running, label=f"{model_name}")
        axs_ee_error[1].plot(ee_pos_error_cumulative, label=f"{model_name}")

        axs_base_error[0, 0].plot(base_pos_error_running, label=f"{model_name}")
        axs_base_error[1, 0].plot(base_pos_error_cumulative, label=f"{model_name}")

        axs_base_error[0, 1].plot(base_theta_error_running, label=f"{model_name}")
        axs_base_error[1, 1].plot(base_theta_error_cumulative, label=f"{model_name}")

        # plot max and total errors
        axs_ee_error[2].barh(
            list(model_errors_max_ee.keys()), list(model_errors_max_ee.values())
        )
        axs_ee_error[3].barh(
            list(model_errors_total_ee.keys()), list(model_errors_total_ee.values())
        )
        axs_base_error[2, 0].barh(
            list(model_errors_max_base.keys()), list(model_errors_max_base.values())
        )
        axs_base_error[3, 0].barh(
            list(model_errors_total_base.keys()),
            list(model_errors_total_base.values()),
        )
        axs_base_error[2, 1].barh(
            list(model_errors_max_base_theta.keys()),
            list(model_errors_max_base_theta.values()),
        )
        axs_base_error[3, 1].barh(
            list(model_errors_total_base_theta.keys()),
            list(model_errors_total_base_theta.values()),
        )

        for i in range(6):
            axs[i, 0].legend(fontsize=8)
            axs[i, 1].legend(fontsize=8)

        for i in range(4):
            axs_ee_error[i].legend(fontsize=8)
            axs_base_error[i, 0].legend(fontsize=8)
            axs_base_error[i, 1].legend(fontsize=8)

        if wandb_logging:
            wandb.log(
                {
                    f"sys_id_extra_eval/{traj_id}/plot": wandb.Image(fig),
                    f"sys_id_extra_eval/{traj_id}/ee_error_plot": wandb.Image(
                        fig_ee_error
                    ),
                    f"sys_id_extra_eval/{traj_id}/base_error_plot": wandb.Image(
                        fig_base_error
                    ),
                }
            )

    # calculate rmse across all eval trajectories
    for state_label in state_labels:
        state_idx = state_labels.index(state_label)
        extra_eval_metrics[f"{state_label}_rmse"] = jnp.sqrt(
            extra_eval_metrics[f"{state_label}_rmse"] / len(eval_trajectories)
        )
    extra_eval_metrics["base_pos_rmse"] = jnp.sqrt(
        extra_eval_metrics["base_pos_rmse"] / len(eval_trajectories)
    )
    extra_eval_metrics["theta_rmse"] = jnp.sqrt(
        extra_eval_metrics["theta_rmse"] / len(eval_trajectories)
    )
    extra_eval_metrics["ee_pos_rmse"] = jnp.sqrt(
        extra_eval_metrics["ee_pos_rmse"] / len(eval_trajectories)
    )

    # log extra eval metrics
    if wandb_logging:
        for step in range(step_range):
            step_extra_eval_metrics = {
                f"sys_id_extra_eval/{state_label}_rmse": float(
                    extra_eval_metrics[f"{state_label}_rmse"][step]
                )
                for state_label in state_labels
            }
            step_extra_eval_metrics.update(
                {
                    "sys_id_extra_eval/base_pos_rmse": float(
                        extra_eval_metrics["base_pos_rmse"][step]
                    ),
                    "sys_id_extra_eval/theta_rmse": float(
                        extra_eval_metrics["theta_rmse"][step]
                    ),
                    "sys_id_extra_eval/ee_pos_rmse": float(
                        extra_eval_metrics["ee_pos_rmse"][step]
                    ),
                    "sys_id_extra_eval/step": step,
                }
            )
            wandb.log(step_extra_eval_metrics)


def run_spot_system_id(
    session_paths: List[str],
    params_to_use: Dict[str, bool],
    weights: jnp.array,
    encode_angle: bool = False,
    num_offline_collected_transitions: int = 5000,
    num_sim_fitting_steps: int = 40_000,
    rng_key: jnp.array = jax.random.PRNGKey(0),
    test_data_ratio: float = 0.1,
    wandb_logging: bool = True,
) -> Dict:

    for key, value in params_to_use.items():
        assert key in ["alpha", "beta_pos", "beta_vel", "gamma"], "No such param"

    def prepare_spot_data(
        dataset: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        window_size=10,
        encode_angles: bool = False,
        angle_idx: int = 2,
        action_delay_base: int = 0,
        action_delay_ee: int = 0,
    ):
        x, u, _ = dataset

        u = delay_and_stack_spot_actions(
            u=u,
            action_stacking=False,
            action_delay_base=action_delay_base,
            action_delay_ee=action_delay_ee,
        )

        x = x.at[:, angle_idx].set((x[:, angle_idx] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        if encode_angles:
            x = encode_angles_fn(x, angle_idx=angle_idx)
        x_strides = get_trajectory_windows(x, window_size)
        u_strides = get_trajectory_windows(u, window_size)
        return x_strides, u_strides

    def simulate_traj(x0: jnp.array, u_traj, params, num_steps: int) -> jnp.array:
        pred_traj = [x0]
        x = x0
        for i in range(num_steps):
            x_pred = step_vmap(
                x,
                u_traj[..., i, :],
                SpotParams(
                    **params["spot_model"],
                ),
            )
            pred_traj.append(x_pred)
            x = x_pred
        pred_traj = jnp.stack(pred_traj, axis=-2)
        assert pred_traj.shape[-2:] == (num_steps + 1, x0.shape[-1])
        return pred_traj

    def trajecory_diff(
        traj1: jnp.array, traj2: jnp.array, angle_idx: int = 2
    ) -> jnp.array:
        """Compute the difference between two trajectories. Accounts for angles on the circle."""
        assert traj1.shape == traj2.shape
        # compute diff between predicted and real trajectory
        diff = traj1 - traj2

        # special treatment for theta (i.e. shortest distance between angles on the circle)
        theta_diff = angle_diff(traj1[..., angle_idx], traj2[..., angle_idx])
        diff = jnp.concatenate(
            [diff[..., :angle_idx], theta_diff[..., None], diff[..., angle_idx + 1 :]],
            axis=-1,
        )
        assert diff.shape == traj1.shape
        return diff

    def loss_fn(
        params,
        x_strided,
        u_strided,
        num_steps_ahead: int = 3,
        weights: jnp.array = None,
    ):
        assert x_strided.shape[-2] > num_steps_ahead

        pred_traj = simulate_traj(
            x_strided[..., 0, :], u_strided, params, num_steps_ahead
        )
        pred_traj = pred_traj[
            ..., 1:, :
        ]  # remove first state (which is the initial state)

        # compute diff between predicted and real trajectory
        real_traj = x_strided[..., 1 : 1 + num_steps_ahead, :]
        diff = trajecory_diff(real_traj, pred_traj)

        pred_dist = tfd.Normal(
            jnp.zeros_like(params["noise_log_std"]), jnp.exp(params["noise_log_std"])
        )
        log_probs = pred_dist.log_prob(diff)

        if weights is not None:
            log_probs = log_probs * weights

        loss = -jnp.mean(log_probs)
        return loss

    def plot_trajectory_comparison(real_traj, sim_traj):
        if encode_angle:

            def decode_angle(obs):
                sin_theta, cos_theta = obs[2], obs[3]
                theta = jnp.arctan2(sin_theta, cos_theta)
                theta = jnp.expand_dims(theta, axis=-1)
                new_obs = jnp.concatenate([obs[:2], theta, obs[4:]], axis=-1)
                return new_obs

            real_traj = jax.vmap(decode_angle)(real_traj)
            sim_traj = jax.vmap(decode_angle)(sim_traj)
        assert (
            real_traj.shape == sim_traj.shape
            and real_traj.shape[-1] == 12
            and real_traj.ndim == 2
        )

        fig, axes = plt.subplots(nrows=6, ncols=2)
        t = np.arange(sim_traj.shape[0]) / 10.0
        labels = [
            "base_pos_x",
            "base_pos_y",
            "theta",
            "base_vel_x",
            "base_vel_y",
            "theta_vel",
            "ee_pos_x",
            "ee_pos_y",
            "ee_pos_z",
            "ee_vel_x",
            "ee_vel_y",
            "ee_vel_z",
        ]

        for i in range(6):
            axes[i, 0].plot(t, real_traj[:, i], label="real", color="green")
            axes[i, 0].plot(t, sim_traj[:, i], label="sim", color="orange")
            axes[i, 0].set_title(labels[i])
            axes[i, 0].set_xlabel("time (sec)")
            axes[i, 0].set_ylabel(labels[i])
            axes[i, 0].legend()
            axes[i, 1].plot(t, real_traj[:, i + 6], label="real", color="green")
            axes[i, 1].plot(t, sim_traj[:, i + 6], label="sim", color="orange")
            axes[i, 1].set_title(labels[i + 6])
            axes[i, 1].set_xlabel("time (sec)")
            axes[i, 1].set_ylabel(labels[i + 6])
            axes[i, 1].legend()

        return fig

    def eval(params, x_eval, u_eval, log_plots: bool = False):
        traj_pred = simulate_traj(x_eval[..., 0, :], u_eval, params, num_steps=10)
        diff = trajecory_diff(traj_pred, x_eval)

        base_pos_dist = jnp.mean(jnp.linalg.norm(diff[..., :, :2], axis=-1), axis=0)
        theta_diff = jnp.mean(jnp.abs(diff[..., 2]), axis=0)
        base_vel_dist = jnp.mean(jnp.linalg.norm(diff[..., :, 3:5], axis=-1), axis=0)
        ee_pos_dist = jnp.mean(jnp.linalg.norm(diff[..., :, 6:9], axis=-1), axis=0)
        ee_vel_dist = jnp.mean(jnp.linalg.norm(diff[..., :, 9:11], axis=-1), axis=0)

        metrics = {
            "sys_id_eval/base_pos_dist_1": base_pos_dist[1],
            "sys_id_eval/base_pos_dist_5": base_pos_dist[5],
            "sys_id_eval/base_pos_dist_10": base_pos_dist[10],
            "sys_id_eval/base_pos_dist_30": base_pos_dist[30],
            "sys_id_eval/base_pos_dist_60": base_pos_dist[60],
            "sys_id_eval/theta_diff_1": theta_diff[1],
            "sys_id_eval/theta_diff_5": theta_diff[5],
            "sys_id_eval/theta_diff_10": theta_diff[10],
            "sys_id_eval/theta_diff_30": theta_diff[30],
            "sys_id_eval/theta_diff_60": theta_diff[60],
            "sys_id_eval/base_vel_dist_1": base_vel_dist[1],
            "sys_id_eval/base_vel_dist_5": base_vel_dist[5],
            "sys_id_eval/base_vel_dist_10": base_vel_dist[10],
            "sys_id_eval/base_vel_dist_30": base_vel_dist[30],
            "sys_id_eval/base_vel_dist_60": base_vel_dist[60],
            "sys_id_eval/ee_pos_dist_1": ee_pos_dist[1],
            "sys_id_eval/ee_pos_dist_5": ee_pos_dist[5],
            "sys_id_eval/ee_pos_dist_10": ee_pos_dist[10],
            "sys_id_eval/ee_pos_dist_30": ee_pos_dist[30],
            "sys_id_eval/ee_pos_dist_60": ee_pos_dist[60],
            "sys_id_eval/ee_vel_dist_1": ee_vel_dist[1],
            "sys_id_eval/ee_vel_dist_5": ee_vel_dist[5],
            "sys_id_eval/ee_vel_dist_10": ee_vel_dist[10],
            "sys_id_eval/ee_vel_dist_30": ee_vel_dist[30],
            "sys_id_eval/ee_vel_dist_60": ee_vel_dist[60],
        }
        if log_plots:
            plots = {
                f"sys_id_eval/trajectory_comparison_{i}": plot_trajectory_comparison(
                    x_eval[i], traj_pred[i]
                )
                for i in [1, 200, 800, 1800, 2600, 3000, 4300, 5500]
            }
            return {**metrics, **plots}
        else:
            return metrics

    # prepare keys
    key_init, key_train, key_data = jax.random.split(rng_key, 3)

    # optimization settings
    angle_idx = 2
    batch_size = 64
    window_size = 11
    num_steps_ahead = 3
    dataset_pre = _load_spot_datasets(session_paths)
    assert weights.shape == (13,) if encode_angle else (12,)

    # prepare data
    datasets = list(
        map(
            partial(
                prepare_spot_data,
                window_size=window_size,
                encode_angles=encode_angle,
            ),
            dataset_pre,
        )
    )
    datasets = [
        datasets[i] for i in jax.random.permutation(key_data, jnp.arange(len(datasets)))
    ]
    x_, u_ = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets))

    # split into train and test
    # note: Transitions have to be consecutive in time, since we are doing multi-step prediction
    num_samples_train = int(num_offline_collected_transitions * (1 - test_data_ratio))
    num_samples_test = int(num_offline_collected_transitions * test_data_ratio)
    assert num_samples_train + num_samples_test < x_.shape[0], "Not enough data"
    x_train, u_train = x_[:num_samples_train], u_[:num_samples_train]
    x_test, u_test = (
        x_[num_samples_train : num_samples_test + num_samples_train],
        u_[num_samples_train : num_samples_test + num_samples_train],
    )

    # confirm shape
    print(
        "x_train.shape",
        x_train.shape,
        "y_train.shape",
        x_train.shape,  # note: y_train is same as x_train here
        "x_eval.shape",
        x_test.shape,
        "y_eval.shape",
        x_test.shape,  # note: y_eval is same as x_eval here
    )

    # initialize dynamics model
    dynamics = SpotDynamicsModel(dt=1.0 / 10.0, encode_angle=encode_angle)
    step_vmap = jax.vmap(dynamics.next_step, in_axes=(0, 0, None), out_axes=0)

    # initialize parameters
    (
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
        k13,
        k14,
        k15,
        k16,
        k17,
        k18,
        k19,
        k20,
        k21,
        k22,
        k23,
        k24,
    ) = jax.random.split(key_init, 24)

    params_alpha = {
        "alpha_base_1": jax.random.uniform(k1, minval=0.0, maxval=1.0),
        "alpha_base_2": jax.random.uniform(k2, minval=0.0, maxval=1.0),
        "alpha_base_3": jax.random.uniform(k3, minval=0.0, maxval=1.0),
        "alpha_ee_1": jax.random.uniform(k4, minval=0.0, maxval=1.0),
        "alpha_ee_2": jax.random.uniform(k5, minval=0.0, maxval=1.0),
        "alpha_ee_3": jax.random.uniform(k6, minval=0.0, maxval=1.0),
    }

    params_betapos = {
        "beta_base_1": jax.random.uniform(k7, minval=-0.5, maxval=0.5),
        "beta_base_2": jax.random.uniform(k8, minval=-0.5, maxval=0.5),
        "beta_base_3": jax.random.uniform(k9, minval=-0.5, maxval=0.5),
        "beta_ee_1": jax.random.uniform(k13, minval=-0.5, maxval=0.5),
        "beta_ee_2": jax.random.uniform(k14, minval=-0.5, maxval=0.5),
        "beta_ee_3": jax.random.uniform(k15, minval=-0.5, maxval=0.5),
    }

    params_betavel = {
        "beta_base_4": jax.random.uniform(k10, minval=-0.5, maxval=0.5),
        "beta_base_5": jax.random.uniform(k11, minval=-0.5, maxval=0.5),
        "beta_base_6": jax.random.uniform(k12, minval=-0.5, maxval=0.5),
        "beta_ee_4": jax.random.uniform(k16, minval=-0.5, maxval=0.5),
        "beta_ee_5": jax.random.uniform(k17, minval=-0.5, maxval=0.5),
        "beta_ee_6": jax.random.uniform(k18, minval=-0.5, maxval=0.5),
    }

    params_gamma = {
        "gamma_base_1": jax.random.uniform(k19, minval=0.0, maxval=1.0),
        "gamma_base_2": jax.random.uniform(k20, minval=0.0, maxval=1.0),
        "gamma_base_3": jax.random.uniform(k21, minval=0.0, maxval=1.0),
        "gamma_ee_1": jax.random.uniform(k22, minval=0.0, maxval=1.0),
        "gamma_ee_2": jax.random.uniform(k23, minval=0.0, maxval=1.0),
        "gamma_ee_3": jax.random.uniform(k24, minval=0.0, maxval=1.0),
    }

    # check which params to use
    params_spot_model = {}
    if params_to_use["alpha"]:
        params_spot_model.update(params_alpha)
    if params_to_use["beta_pos"]:
        params_spot_model.update(params_betapos)
    if params_to_use["beta_vel"]:
        params_spot_model.update(params_betavel)
    if params_to_use["gamma"]:
        params_spot_model.update(params_gamma)

    params = {
        "spot_model": params_spot_model,
        "noise_log_std": -1.0 * jnp.ones((num_steps_ahead, 13 if encode_angle else 12)),
    }

    # prepare optimizer
    optim = optax.adam(1e-3)
    opt_state = optim.init(params)

    @jax.jit
    def step(params, opt_state, key: jax.random.PRNGKey):
        idx_batch = jax.random.choice(key, x_train.shape[0], shape=(batch_size,))
        x_batch, u_batch = x_train[idx_batch], u_train[idx_batch]
        loss, grads = jax.value_and_grad(loss_fn)(
            params, x_batch, u_batch, weights=weights
        )
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # run optimization
    for i in range(num_sim_fitting_steps):
        key_train, subkey = jax.random.split(key_train)
        loss, params, opt_state = step(params, opt_state, subkey)

        if i % 1000 == 0:
            loss_test = loss_fn(params, x_test, u_test)
            metrics_eval = eval(params, x_test, u_test, log_plots=True)
            if wandb_logging:
                wandb.log(
                    {"iter": i, "loss": loss, "loss_test": loss_test, **metrics_eval}
                )
            print(f"Iter {i}, loss: {loss}, test loss: {loss_test}")

    # fetch results
    spot_learned_params = params["spot_model"]
    spot_learned_observation_noise_std = 0.1 * jnp.exp(
        jnp.array(params["noise_log_std"][0])
    )

    return spot_learned_params, spot_learned_observation_noise_std


def execute_spot_system_id(
    random_seed: int,
    num_offline_collected_transitions: int,
    test_data_ratio: float,
    wandb_logging: bool,
    num_sim_fitting_steps: int = 40_000,
    default_option: str = "alpha_beta_vel",
) -> Dict:

    # set options
    options = {
        "alpha_only": [True, False, False, False],
        "gamma_only": [False, False, False, True],
        "alpha_beta_pos": [True, True, False, False],
        "alpha_beta_vel": [True, False, True, False],
        "alpha_gamma": [True, False, False, True],
        "alpha_beta_pos_beta_vel": [True, True, True, False],
        "alpha_beta_pos_gamma": [True, True, False, True],
        "alpha_beta_vel_gamma": [True, False, True, True],
        "all": [True, True, True, True],
    }

    # filter options
    options = (
        {key: value for key, value in options.items() if key == default_option}
        if default_option
        else options
    )

    # set parameters
    encode_angle = False
    weights = None

    # load data dirs
    DATA_DIR = "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data"
    recordings_dirs = [
        os.path.join(DATA_DIR, "recordings_spot_v0"),
        os.path.join(DATA_DIR, "recordings_spot_v1"),
        os.path.join(DATA_DIR, "recordings_spot_v2"),
        os.path.join(DATA_DIR, "recordings_spot_v3"),
        os.path.join(DATA_DIR, "recordings_spot_v4"),
    ]
    session_paths = sorted(
        [
            file
            for dir_path in recordings_dirs
            for file in glob.glob(os.path.join(dir_path, "*.pickle"))
        ]
    )

    # run system id
    saved_params = {}
    for key, value in options.items():
        params = run_spot_system_id(
            session_paths=session_paths,
            params_to_use={
                "alpha": value[0],
                "beta_pos": value[1],
                "beta_vel": value[2],
                "gamma": value[3],
            },
            weights=weights,
            encode_angle=encode_angle,
            num_offline_collected_transitions=num_offline_collected_transitions,
            rng_key=jax.random.PRNGKey(random_seed),
            test_data_ratio=test_data_ratio,
            num_sim_fitting_steps=num_sim_fitting_steps,
            wandb_logging=wandb_logging,
        )
        saved_params[key] = params

    pprint(saved_params)
    return saved_params[default_option]


def main(args):
    # set parameters
    random_seed = args.random_seed
    num_offline_collected_transitions = args.num_offline_collected_transitions
    test_data_ratio = args.test_data_ratio
    project_name = args.project_name
    wandb_logging = args.wandb_logging

    # initialize wandb
    if wandb_logging:
        wandb.init(
            project=project_name,
            config={
                "num_offline_collected_transitions": num_offline_collected_transitions,
                "test_data_ratio": test_data_ratio,
                "random_seed": random_seed,
            },
        )

    return execute_spot_system_id(
        random_seed=random_seed,
        num_offline_collected_transitions=num_offline_collected_transitions,
        test_data_ratio=test_data_ratio,
        wandb_logging=wandb_logging,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # prevent TF from reserving all GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # parameters
    parser.add_argument("--random_seed", type=int, default=922852)
    parser.add_argument("--project_name", type=str, default="spot_sysrtem_id")
    parser.add_argument("--num_offline_collected_transitions", type=int, default=5_000)
    parser.add_argument("--test_data_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_logging", type=bool, default=True)
    args = parser.parse_args()
    _, _ = main(args)
