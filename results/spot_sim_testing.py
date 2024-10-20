import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Union, Optional, Tuple, Dict
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import random, vmap
from jaxtyping import PyTree

from sim_transfer.sims.util import angle_diff, decode_angles, encode_angles
from sim_transfer.sims.dynamics_models import SpotParams, SpotDynamicsModel
from sim_transfer.sims.simulators import SpotSim

############################################ Functions ############################################


def simulate_spot(
    file_path,
    spot_models: Dict[str, SpotParams],
    dt=1.0 / 10.0,
    horizon=None,
    show_plot=True,
    save_path=None,
    action_delay_base: int = 0,
    action_delay_ee: int = 0,
    plot_error_metrics: bool = False,
    start_idx: Optional[int] = None,
    steps: Optional[int] = None,
):

    # load measured data
    with open(file_path, "rb") as f:
        prev_states_jax, u_jax, next_states_jax = pickle.load(f)
        prev_states, u, next_states = (
            np.array(prev_states_jax),
            np.array(u_jax),
            np.array(next_states_jax),
        )
        if start_idx is not None:
            prev_states, u, next_states = (
                prev_states[start_idx:],
                u[start_idx:],
                next_states[start_idx:],
            )
        if steps is not None:
            prev_states, u, next_states = (
                prev_states[:steps],
                u[:steps],
                next_states[:steps],
            )

    if horizon is None:
        horizon = len(u)

    # apply action delay
    u_base, u_ee = u[:, :3], u[:, 3:]
    if action_delay_base > 0:
        u_delayed_start = jnp.zeros_like(u_base[:action_delay_base])
        u_delayed_base = jnp.concatenate([u_delayed_start, u_base[:-action_delay_base]])
        assert u_delayed_base.shape == u_base.shape, "Base action delay failed"
        u_base = u_delayed_base

    if action_delay_ee > 0:
        u_delayed_start = jnp.zeros_like(u_ee[:action_delay_ee])
        u_delayed_ee = jnp.concatenate([u_delayed_start, u_ee[:-action_delay_ee]])
        assert u_delayed_ee.shape == u_ee.shape, "EE action delay failed"
        u_ee = u_delayed_ee
    u = jnp.concatenate([u_base, u_ee], axis=1)

    # setup spot model
    spot = SpotDynamicsModel(
        dt=dt,
        encode_angle=False,
        input_in_local_frame=True,
    )

    spot_sim = SpotSim(encode_angle=False)

    # plot measured data
    labels = [
        "base_x",
        "base_y",
        "base_theta",
        "base_vx",
        "base_vy",
        "base_vtheta",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_vx",
        "ee_vy",
        "ee_vz",
    ]
    fig, axs = plt.subplots(6, 2)
    if plot_error_metrics:
        fig_error_base, axs_error_base = plt.subplots(6, 2)
        fig_error_ee, axs_error_ee = plt.subplots(6, 2)
        fig_error_running, axs_error_running = plt.subplots(6, 4)
        fig_ee_pos_error, axs_ee_pos_error = plt.subplots(4, 1)

    for i in range(6):
        axs[i][0].plot(next_states[:, i], label="measured", color="green")
        axs[i][0].set_title(labels[i])
        axs[i][1].plot(next_states[:, i + 6], label="measured", color="green")
        axs[i][1].set_title(labels[i + 6])
        if i in [3, 4, 5]:
            if i == 5:
                axs[i][0].set_ylabel("rad/s")
                axs[i][1].set_ylabel("m/s")
            else:
                axs[i][0].set_ylabel("m/s")
                axs[i][1].set_ylabel("m/s")
        else:
            if i == 2:
                axs[i][0].set_ylabel("rad")
                axs[i][1].set_ylabel("m")
            else:
                axs[i][0].set_ylabel("m")
                axs[i][1].set_ylabel("m")

    # add labels for errors
    if plot_error_metrics:
        for i in range(6):
            axs_error_running[i][0].set_title(f"{labels[i]} error")
            axs_error_running[i][1].set_title(f"{labels[i + 6]} error")
            axs_error_running[i][2].set_title(f"{labels[i]} cumulated error")
            axs_error_running[i][3].set_title(f"{labels[i + 6]} cumulated error")
            axs_error_base[i][0].set_title(f"{labels[i]} max error")
            axs_error_base[i][1].set_title(f"{labels[i]} cumulated error")
            axs_error_ee[i][0].set_title(f"{labels[i + 6]} max error")
            axs_error_ee[i][1].set_title(f"{labels[i + 6]} cumulated error")
            if i in [3, 4, 5]:
                if i == 5:
                    axs_error_running[i][0].set_ylabel("rad/s")
                    axs_error_running[i][1].set_ylabel("m/s")
                    axs_error_running[i][2].set_ylabel("rad/s")
                    axs_error_running[i][3].set_ylabel("m/s")
                    axs_error_base[i][0].set_ylabel("rad/s")
                    axs_error_base[i][1].set_ylabel("rad/s")
                    axs_error_ee[i][0].set_ylabel("m/s")
                    axs_error_ee[i][1].set_ylabel("m/s")
                else:
                    axs_error_running[i][0].set_ylabel("m/s")
                    axs_error_running[i][1].set_ylabel("m/s")
                    axs_error_running[i][2].set_ylabel("m/s")
                    axs_error_running[i][3].set_ylabel("m/s")
                    axs_error_base[i][0].set_ylabel("m/s")
                    axs_error_base[i][1].set_ylabel("m/s")
                    axs_error_ee[i][0].set_ylabel("m/s")
                    axs_error_ee[i][1].set_ylabel("m/s")
            else:
                if i == 2:
                    axs_error_running[i][0].set_ylabel("rad")
                    axs_error_running[i][1].set_ylabel("m")
                    axs_error_running[i][2].set_ylabel("rad")
                    axs_error_running[i][3].set_ylabel("m")
                    axs_error_base[i][0].set_ylabel("rad")
                    axs_error_base[i][1].set_ylabel("rad")
                    axs_error_ee[i][0].set_ylabel("m")
                    axs_error_ee[i][1].set_ylabel("m")
                else:
                    axs_error_running[i][0].set_ylabel("m")
                    axs_error_running[i][1].set_ylabel("m")
                    axs_error_running[i][2].set_ylabel("m")
                    axs_error_running[i][3].set_ylabel("m")
                    axs_error_base[i][0].set_ylabel("m")
                    axs_error_base[i][1].set_ylabel("m")
                    axs_error_ee[i][0].set_ylabel("m")
                    axs_error_ee[i][1].set_ylabel("m")

        axs_ee_pos_error[0].set_title("ee pos error")
        axs_ee_pos_error[0].set_ylabel("m")
        axs_ee_pos_error[1].set_title("ee pos error cumulated running")
        axs_ee_pos_error[1].set_ylabel("m")
        axs_ee_pos_error[2].set_title("ee pos error max")
        axs_ee_pos_error[2].set_ylabel("m")
        axs_ee_pos_error[3].set_title("ee pos error cumulated")
        axs_ee_pos_error[3].set_ylabel("m")

    # plot simulated data
    cmap = plt.get_cmap("Dark2") if len(spot_models) <= 8 else plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(spot_models)))

    if plot_error_metrics:
        max_error_dict = {}
        cumulative_error_dict = {}
        ee_max_error_dict = {}
        ee_cum_error_dict = {}

    for color, spot_model_name, spot_model in zip(
        colors, spot_models.keys(), spot_models.values()
    ):
        params = SpotParams(**spot_model)

        # simulate data
        x = jnp.zeros(12)
        x_traj = jnp.zeros([horizon, 12])
        x = x.at[0:12].set(prev_states[0])
        for h in range(horizon):
            u_curr = jnp.array(u[h])
            use_dynamics = False
            if use_dynamics:
                x = spot.next_step(x, u_curr, params)
                x_traj = x_traj.at[h, ...].set(x)

            else:
                # fuse state and action
                x_full = jnp.concatenate((x, u_curr), axis=0)
                x_full = x_full.reshape(1, x_full.size)
                x = spot_sim.evaluate_sim(x_full, params)
                # extract state
                x = x[0, :12]
                x = x.reshape(12)
                x_traj = x_traj.at[h, ...].set(x)

        # plot simulated data
        for i in range(6):
            axs[i][0].plot(
                x_traj[:, i],
                label=f"simulated_{spot_model_name}",
                color=color,
                linestyle="dashed",
                alpha=0.9,
            )
            axs[i][1].plot(
                x_traj[:, i + 6],
                label=f"simulated_{spot_model_name}",
                color=color,
                linestyle="dashed",
                alpha=0.9,
            )

        # calculate error metrics
        if plot_error_metrics:
            trajectory_error = jnp.abs(next_states - x_traj)
            cumulatitive_error = jnp.sum(trajectory_error, axis=0)
            max_error = jnp.max(trajectory_error, axis=0)
            max_error_dict[spot_model_name] = max_error
            cumulative_error_dict[spot_model_name] = cumulatitive_error
            cumulative_error_per_step = jnp.zeros_like(trajectory_error)
            for i in range(horizon):
                cumulative_error_per_step = cumulative_error_per_step.at[i, ...].set(
                    jnp.sum(trajectory_error[: i + 1], axis=0)
                )

            # plot trajectory and cumulative error
            for i in range(6):
                axs_error_running[i][0].plot(
                    trajectory_error[:, i],
                    label=f"error_{spot_model_name}",
                    color=color,
                    linestyle="dashed",
                    alpha=0.9,
                )
                axs_error_running[i][1].plot(
                    trajectory_error[:, i + 6],
                    label=f"error_{spot_model_name}",
                    color=color,
                    linestyle="dashed",
                    alpha=0.9,
                )
                axs_error_running[i][2].plot(
                    cumulative_error_per_step[:, i],
                    label=f"cumulative_error_{spot_model_name}",
                    color=color,
                    linestyle="dashed",
                    alpha=0.9,
                )
                axs_error_running[i][3].plot(
                    cumulative_error_per_step[:, i + 6],
                    label=f"cumulative_error_{spot_model_name}",
                    color=color,
                    linestyle="dashed",
                    alpha=0.9,
                )

            # calculate ee pos error. ee pos are idx 6, 7, 8
            ee_error_running = jnp.sqrt(jnp.sum(trajectory_error[:, 6:9] ** 2, axis=1))
            ee_error_cum = jnp.sum(ee_error_running, axis=0)
            ee_error_max = jnp.max(ee_error_running, axis=0)
            ee_max_error_dict[spot_model_name] = ee_error_max
            ee_cum_error_dict[spot_model_name] = ee_error_cum
            ee_cum_error_per_step = jnp.cumsum(ee_error_running, axis=0)

            # plot ee pos error
            axs_ee_pos_error[0].plot(
                ee_error_running,
                label=f"ee_error_{spot_model_name}",
                color=color,
                linestyle="dashed",
                alpha=0.9,
            )

            axs_ee_pos_error[1].plot(
                ee_cum_error_per_step,
                label=f"cumulative_error_{spot_model_name}",
                color=color,
                linestyle="dashed",
                alpha=0.9,
            )

        print(f"Simulated for {spot_model_name}")

    # plot max and cumulative errors
    if plot_error_metrics:

        # overall
        for i in range(6):
            axs_error_base[i][0].barh(
                list(max_error_dict.keys()),
                [max_error[i] for max_error in max_error_dict.values()],
                color=colors,
            )
            axs_error_base[i][1].barh(
                list(cumulative_error_dict.keys()),
                [
                    cumulative_error[i]
                    for cumulative_error in cumulative_error_dict.values()
                ],
                color=colors,
            )
            axs_error_ee[i][0].barh(
                list(max_error_dict.keys()),
                [max_error[i + 6] for max_error in max_error_dict.values()],
                color=colors,
            )
            axs_error_ee[i][1].barh(
                list(cumulative_error_dict.keys()),
                [
                    cumulative_error[i + 6]
                    for cumulative_error in cumulative_error_dict.values()
                ],
                color=colors,
            )

        # ee pos
        axs_ee_pos_error[2].barh(
            list(ee_max_error_dict.keys()),
            [ee_error_max for ee_error_max in ee_max_error_dict.values()],
            color=colors,
        )
        axs_ee_pos_error[3].barh(
            list(ee_cum_error_dict.keys()),
            [ee_error_cum for ee_error_cum in ee_cum_error_dict.values()],
            color=colors,
        )

    fig.legend(["measured", *spot_models.keys()], loc="center left")
    if plot_error_metrics:
        fig_error_running.legend([*spot_models.keys()], loc="center left")
        fig_ee_pos_error.legend([*spot_models.keys()], loc="center left")

    if show_plot:
        plt.show()

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path + "_trajectory")
        if plot_error_metrics:
            fig_error_base.savefig(save_path + "_base_error")
            fig_error_ee.savefig(save_path + "_ee_error")
            fig_error_running.savefig(save_path + "_running_error")
            fig_ee_pos_error.savefig(save_path + "_ee_pos_error")


def simulate_spot_range(
    file_path,
    param: str = "alpha",
    range: Tuple[float, float] = (0.0, 1.0),
    steps: float = 0.1,
    dt=1.0 / 10.0,
    horizon=None,
    show_plot=True,
    save_path=None,
):
    import numpy as np

    assert param in [
        "alpha",
        "beta_base",
        "beta_ee",
        "gamma",
    ], "param should be one of ['alpha', 'beta_base', 'beta_ee', 'gamma']"
    spot_models = {}
    amount = int((range[1] - range[0]) / steps)
    for p in np.linspace(range[0], range[1], amount):
        p = round(p, 3)
        if param == "alpha":
            spot_model = {
                "alpha_base_1": p,
                "alpha_base_2": p,
                "alpha_base_3": p,
                "alpha_ee_1": p,
                "alpha_ee_2": p,
                "alpha_ee_3": p,
            }
        elif param == "beta_base":
            spot_model = {
                "beta_base_1": p,
                "beta_base_2": p,
                "beta_base_3": p,
                "beta_base_4": p,
                "beta_base_5": p,
                "beta_base_6": p,
            }
        elif param == "beta_ee":
            spot_model = {
                "beta_ee_1": p,
                "beta_ee_2": p,
                "beta_ee_3": p,
                "beta_ee_4": p,
                "beta_ee_5": p,
                "beta_ee_6": p,
            }
        elif param == "gamma":
            spot_model = {
                "gamma_base_1": p,
                "gamma_base_2": p,
                "gamma_base_3": p,
                "gamma_ee_1": p,
                "gamma_ee_2": p,
                "gamma_ee_3": p,
            }
        spot_models[f"{param}_{p}"] = spot_model

    simulate_spot(file_path, spot_models, dt, horizon, show_plot, save_path)


def plot_params(params):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(20, 10))
    # plot alpha params
    alpha_keys = [key for key in params.keys() if "alpha" in key]
    alpha_values = [params[key] for key in alpha_keys]
    alpha_labels = [key for key in alpha_keys]
    axs[0].bar(alpha_labels, alpha_values)
    axs[0].set_title("Alpha parameters")
    # plot beta params for base
    beta_base_keys = [key for key in params.keys() if "beta_base" in key]
    beta_base_values = [params[key] for key in beta_base_keys]
    beta_base_labels = [key for key in beta_base_keys]
    axs[1].bar(beta_base_labels, beta_base_values)
    axs[1].set_title("Beta base parameters")
    # plot beta params for ee
    beta_ee_keys = [key for key in params.keys() if "beta_ee" in key]
    beta_ee_values = [params[key] for key in beta_ee_keys]
    beta_ee_labels = [key for key in beta_ee_keys]
    axs[2].bar(beta_ee_labels, beta_ee_values)
    axs[2].set_title("Beta ee parameters")
    # plot gamma params
    gamma_keys = [key for key in params.keys() if "gamma" in key]
    gamma_values = [params[key] for key in gamma_keys]
    gamma_labels = [key for key in gamma_keys]
    axs[3].bar(gamma_labels, gamma_values)
    axs[3].set_title("Gamma parameters")

    plt.show()


############################################ Set file path ############################################
file_path = (
    "data/recordings_spot_v0/dataset_learn_jax_20240815-151559_20240816-105027.pickle"
)
# file_path = "data/recordings_spot_v0/dataset_learn_jax_20240819-142443_20240820-101938.pickle"
file_path = "/home/bhoffman/Documents/MT FS24/active-learning-dynamics/data_analysis/learning_data/dataset_learn_jax_test20240830-112255_v1_3.pickle"


# import and define params
from sim_transfer.sims.spot_sim_config import *

spot_models_set_1 = {
    "alpha_set_1": spot_model_alpha_set_1,
    "gamma_set_1": spot_model_gamma_set_1,
    "alpha_betapos_set_1": spot_model_alpha_betapos_set_1,
    "alpha_betavel_set_1": spot_model_alpha_betavel_set_1,
    "alpha_gamma_set_1": spot_model_alpha_gamma_set_1,
    "alpha_betapos_gamma_set_1": spot_model_alpha_betapos_gamma_set_1,
    "all_set_1": spot_model_all_set_1,
}

spot_models_set_2 = {
    "alpha_set_2": spot_model_alpha_set_2,
    "gamma_set_2": spot_model_gamma_set_2,
    "alpha_betapos_set_2": spot_model_alpha_betapos_set_2,
    "alpha_betavel_set_2": spot_model_alpha_betavel_set_2,
    "alpha_gamma_new": spot_model_alpha_gamma_set_2,
    "alpha_betapos_gamma_set_2": spot_model_alpha_betapos_gamma_set_2,
    "all_set_2": spot_model_all_set_2,
}

spot_models_set_3 = {
    "alpha_set_3": spot_model_alpha_set_3,
    "gamma_set_3": spot_model_gamma_set_3,
    "alpha_beta_pos_set_3": spot_model_alpha_beta_pos_set_3,
    "alpha_beta_pos_beta_vel_set_3": spot_model_alpha_beta_pos_beta_vel_set_3,
    "alpha_beta_pos_gamma_set_3": spot_model_alpha_beta_pos_gamma_set_3,
    "alpha_beta_vel_set_3": spot_model_alpha_beta_vel_set_3,
    "alpha_beta_vel_gamma_set_3": spot_model_alpha_beta_vel_gamma_set_3,
    "alpha_gamma_set_3": spot_model_alpha_gamma_set_3,
    "all_set_3": spot_model_all_set_3,
}

spot_models_set_5 = {
    "alpha_set_5": spot_model_alpha_set_5,
    "gamma_set_5": spot_model_gamma_set_5,
    "alpha_beta_pos_set_5": spot_model_alpha_beta_pos_set_5,
    "alpha_beta_pos_beta_vel_set_5": spot_model_alpha_beta_pos_beta_vel_set_5,
    "alpha_beta_pos_gamma_set_5": spot_model_alpha_beta_pos_gamma_set_5,
    "alpha_beta_vel_set_5": spot_model_alpha_beta_vel_set_5,
    "alpha_beta_vel_gamma_set_5": spot_model_alpha_beta_vel_gamma_set_5,
    "alpha_gamma_set_5": spot_model_alpha_gamma_set_5,
    "all_set_5": spot_model_all_set_5,
}


spot_model_comparison = {
    # "alpha_betavel_set_1": spot_model_alpha_betavel_set_1,
    # "alpha_betavel_set_2": spot_model_alpha_betavel_set_2,
    # "alpha_betavel_gama_set_3": spot_model_alpha_beta_vel_gamma_set_3,
    "alpha_beta_vel_set_5": spot_model_alpha_beta_vel_set_5,

}


############################################ Run Simulation ############################################

simulate_spot(
    file_path=file_path,
    spot_models=spot_model_comparison,
    save_path="results/sim_fitting/testing_new_constraints/output",
    action_delay_base=2,
    action_delay_ee=1,
    plot_error_metrics=True,
    start_idx=0,
    steps=50,
)

"""
Determining best model per set:

set_1: alpha_betavel
set_2: alpha_betavel
set_3: alpha_betavel_gama
set_5: alpha_betavel

Determining best model overall:

"""
