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
from sim_transfer.sims.dynamics_models import SpotDynamicsModel, SpotParams
from sim_transfer.sims.util import angle_diff

from brax.training.types import Transition
import tensorflow_probability.substrates.jax.distributions as tfd


def run_spot_system_id(
    params_to_use: Dict[str, bool],
    weights: jnp.array,
    encode_angle: bool = False,
    session_paths: Union[
        str, List[str]
    ] = "data/recordings_spot_v0/dataset_learn_jax_20240815-151559_20240816-105027.pickle",
) -> Dict:

    for key, value in params_to_use.items():
        assert key in ["alpha", "beta_pos", "beta_vel", "gamma"], "No such param"

    def load_spot_datasets(file_path: Union[str, List[str]]):
        if isinstance(file_path, list):
            dataset = []
            for path in file_path:
                with open(path, "rb") as f:
                    dataset.append(pickle.load(f))
        else:
            with open(file_path, "rb") as f:
                dataset = pickle.load(f)
        return dataset

    def prepare_spot_data(
        dataset: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        window_size=10,
        encode_angles: bool = False,
        angle_idx: int = 2,
        action_delay_base: int = 2,
        action_delay_ee: int = 1,
    ):
        x, u, _ = dataset

        u_base, u_ee = u[:, :3], u[:, 3:]
        if action_delay_base > 0:
            u_delayed_start = jnp.zeros_like(u_base[:action_delay_base])
            u_delayed_base = jnp.concatenate(
                [u_delayed_start, u_base[:-action_delay_base]]
            )
            assert (
                u_delayed_base.shape == u_base.shape
            ), "Something went wrong with the base action delay"
            u_base = u_delayed_base

        if action_delay_ee > 0:
            u_delayed_start = jnp.zeros_like(u_ee[:action_delay_ee])
            u_delayed_ee = jnp.concatenate([u_delayed_start, u_ee[:-action_delay_ee]])
            assert (
                u_delayed_ee.shape == u_ee.shape
            ), "Something went wrong with the ee action delay"
            u_ee = u_delayed_ee

        u = jnp.concatenate([u_base, u_ee], axis=-1)

        x = x.at[:, angle_idx].set((x[:, angle_idx] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        if encode_angles:

            def angle_encode(obs):
                theta = obs[..., angle_idx]
                sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
                sin_theta = jnp.expand_dims(sin_theta, axis=-1)
                cos_theta = jnp.expand_dims(cos_theta, axis=-1)
                encoded_obs = jnp.concatenate(
                    [
                        obs[..., :angle_idx],
                        sin_theta,
                        cos_theta,
                        obs[..., angle_idx + 1 :],
                    ],
                    axis=-1,
                )
                return encoded_obs

            x = jax.vmap(angle_encode)(x)
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
            "base_pos_dist_1": base_pos_dist[1],
            "base_pos_dist_5": base_pos_dist[5],
            "base_pos_dist_10": base_pos_dist[10],
            "base_pos_dist_30": base_pos_dist[30],
            "base_pos_dist_60": base_pos_dist[60],
            "theta_diff_1": theta_diff[1],
            "theta_diff_5": theta_diff[5],
            "theta_diff_10": theta_diff[10],
            "theta_diff_30": theta_diff[30],
            "theta_diff_60": theta_diff[60],
            "base_vel_dist_1": base_vel_dist[1],
            "base_vel_dist_5": base_vel_dist[5],
            "base_vel_dist_10": base_vel_dist[10],
            "base_vel_dist_30": base_vel_dist[30],
            "base_vel_dist_60": base_vel_dist[60],
            "ee_pos_dist_1": ee_pos_dist[1],
            "ee_pos_dist_5": ee_pos_dist[5],
            "ee_pos_dist_10": ee_pos_dist[10],
            "ee_pos_dist_30": ee_pos_dist[30],
            "ee_pos_dist_60": ee_pos_dist[60],
            "ee_vel_dist_1": ee_vel_dist[1],
            "ee_vel_dist_5": ee_vel_dist[5],
            "ee_vel_dist_10": ee_vel_dist[10],
            "ee_vel_dist_30": ee_vel_dist[30],
            "ee_vel_dist_60": ee_vel_dist[60],
        }
        if log_plots:
            plots = {
                f"trajectory_comparison_{i}": plot_trajectory_comparison(
                    x_eval[i], traj_pred[i]
                )
                for i in [1, 200, 800, 1800, 2600, 3000, 4300, 5500]
            }
            return {**metrics, **plots}
        else:
            return metrics

    # prepare keys
    key_init, key_train, key_data = jax.random.split(jax.random.PRNGKey(234234), 3)

    # optimization settings
    angle_idx = 2
    batch_size = 64
    window_size = 11
    test_train_ratio = 0.8
    num_steps_ahead = 3
    dataset_pre = load_spot_datasets(session_paths)
    assert weights.shape == (13,) if encode_angle else (12,)

    if isinstance(dataset_pre, list):
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
            datasets[i]
            for i in jax.random.permutation(key_data, jnp.arange(len(datasets)))
        ]

        # split into train and test
        ratio_int = int(test_train_ratio * len(datasets))
        datasets_train = datasets[:ratio_int]
        datasets_test = datasets[ratio_int:]

        x_train, u_train = map(
            lambda x: jnp.concatenate(x, axis=0), zip(*datasets_train)
        )
        x_test, u_test = map(lambda x: jnp.concatenate(x, axis=0), zip(*datasets_test))
    else:
        # split into train and test
        dataset_train = tuple(
            dataset_pre_arr[: int(test_train_ratio * len(dataset_pre[0]))]
            for dataset_pre_arr in dataset_pre
        )
        dataset_test = tuple(
            dataset_pre_arr[int(test_train_ratio * len(dataset_pre[0])) :]
            for dataset_pre_arr in dataset_pre
        )

        dataset_train = prepare_spot_data(
            dataset_train,
            window_size=window_size,
            encode_angles=encode_angle,
        )
        dataset_test = prepare_spot_data(
            dataset_test,
            window_size=window_size,
            encode_angles=encode_angle,
        )

        x_train, u_train = dataset_train
        x_test, u_test = dataset_test

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
        "noise_log_std": -1. * jnp.ones((num_steps_ahead, 13 if encode_angle else 12)),
    }

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

    run = wandb.init(
        project="system-id-spot",
        entity="benjirer",
    )

    for i in range(40000):
        key_train, subkey = jax.random.split(key_train)
        loss, params, opt_state = step(params, opt_state, subkey)

        if i % 1000 == 0:
            loss_test = loss_fn(params, x_test, u_test)
            metrics_eval = eval(params, x_test, u_test, log_plots=True)
            wandb.log({"iter": i, "loss": loss, "loss_test": loss_test, **metrics_eval})
            print(f"Iter {i}, loss: {loss}, test loss: {loss_test}")

    # pprint(params_to_use)
    # pprint(params)
    return params


if __name__ == "__main__":

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
    encode_angle = False
    # weights = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    weights = None
    session_paths = [
        "data/recordings_spot_v0/dataset_learn_jax_20240815-151559_20240816-105027.pickle",
        "data/recordings_spot_v0/dataset_learn_jax_20240819-141455_20240820-101740.pickle",
        "data/recordings_spot_v0/dataset_learn_jax_20240819-142443_20240820-101938.pickle",
    ]

    saved_params = {}

    for key, value in options.items():
        params = run_spot_system_id(
            params_to_use={
                "alpha": value[0],
                "beta_pos": value[1],
                "beta_vel": value[2],
                "gamma": value[3],
            },
            weights=weights,
            encode_angle=encode_angle,
            session_paths=session_paths,
        )
        saved_params[key] = params

    pprint(saved_params)
