import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, List
import jax.random
import matplotlib.pyplot as plt
from brax.training.types import Transition
import time
import os
from scipy.spatial.transform import Rotation as R

from sim_transfer.sims.spot_sim_config import (
    SPOT_STATE_LENGTH, 
    SPOT_STATE_LENGTH_ENCODED, 
    SPOT_ACTION_LENGTH,
    SPOT_GOAL_LENGTH,
    SPOT_STATE_LABELS, 
    SPOT_STATE_LABELS_ENCODED, 
    SPOT_ACTION_LABELS,
    SPOT_GOAL_LABELS,
    SPOT_ANGLE_IDX,
    )


def encode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """Encodes the angle (theta) as sin(theta) and cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = np.concatenate(
        [
            state[..., :angle_idx],
            np.sin(theta),
            np.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) and cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = jnp.concatenate(
        [
            state[..., :angle_idx],
            jnp.sin(theta),
            jnp.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def encode_angles_spot(state: jnp.array, angle_idx: Union[int, List[int]]) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) and cos(theta) (adapted for Spot robot)"""
    state_shape_org = state.shape[-1]
    len_angle_idx = len(angle_idx) if isinstance(angle_idx, list) else 1
    if isinstance(angle_idx, list):
        angle_idx = [idx + i for i, idx in enumerate(angle_idx)]
        for idx in angle_idx:
            state = encode_angles(state, idx)
    else:
        state = encode_angles(state, angle_idx)
    assert state.shape[-1] == state_shape_org + len_angle_idx
    return state


def decode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = np.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = np.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = jnp.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def decode_angles_spot(state: jnp.array, angle_idx: Union[int, List[int]]) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta) (adapted for Spot robot)"""
    state_shape_org = state.shape[-1]
    len_angle_idx = len(angle_idx) if isinstance(angle_idx, list) else 1
    if isinstance(angle_idx, list):
        for idx in angle_idx:
            state = decode_angles(state, idx)
    else:
        state = decode_angles(state, angle_idx)
    assert state.shape[-1] == state_shape_org - len_angle_idx
    return state


def project_angle(theta: jnp.array) -> jnp.array:
    # make sure angles are in [-pi, pi]
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def angle_diff(theta1: jnp.array, theta2: jnp.array) -> jnp.array:
    # Compute the difference
    diff = theta1 - theta2
    # Normalize to [-pi, pi] range
    diff = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return diff


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = (
        state[..., 0:1],
        state[..., 3 + int(encode_angle) : 4 + int(encode_angle)],
    )
    y_pos, y_vel = (
        state[..., 1:2],
        state[:, 4 + int(encode_angle) : 5 + int(encode_angle)],
    )
    theta = state[..., 2 : 3 + int(encode_angle)]
    new_state = jnp.concatenate(
        [y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle) :]],
        axis=-1,
    )
    assert state.shape == new_state.shape
    return new_state


def plot_rc_trajectory(
    traj: jnp.array,
    actions: Optional[jnp.array] = None,
    pos_domain_size: float = 5,
    show: bool = True,
    encode_angle: bool = False,
):
    """Plots the trajectory of the RC car"""
    if encode_angle:
        traj = decode_angles(traj, 2)

    import matplotlib.pyplot as plt

    scale_factor = 1.5
    if actions is None:
        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8)
        )
    else:
        fig, axes = plt.subplots(
            nrows=2, ncols=4, figsize=(scale_factor * 16, scale_factor * 8)
        )
    axes[0][0].set_xlim(-pos_domain_size, pos_domain_size)
    axes[0][0].set_ylim(-pos_domain_size, pos_domain_size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title("x-y")

    # chaange x -> -y and y -> x
    traj = rotate_coordinates(traj, encode_angle=False)

    # Plot the velocity of the car as vectors
    total_vel = jnp.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(
        traj[0:-1:3, 0],
        traj[0:-1:3, 1],
        traj[0:-1:3, 3],
        traj[0:-1:3, 4],
        total_vel[0:-1:3],
        cmap="jet",
        scale=20,
        headlength=2,
        headaxislength=2,
        headwidth=2,
        linewidth=0.2,
    )

    t = jnp.arange(traj.shape[0]) / 30.0
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel("time")
    axes[0][1].set_ylabel("theta")
    axes[0][1].set_title("theta")

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel("time")
    axes[0][2].set_ylabel("angular velocity")
    axes[0][2].set_title("angular velocity")

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel("time")
    axes[1][0].set_ylabel("total velocity")
    axes[1][0].set_title("velocity")

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel("time")
    axes[1][1].set_ylabel("velocity x")
    axes[1][1].set_title("velocity x")

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel("time")
    axes[1][2].set_ylabel("velocity y")
    axes[1][2].set_title("velocity y")

    if actions is not None:
        # steering
        axes[0][3].plot(t[: actions.shape[0]], actions[:, 0])
        axes[0][3].set_xlabel("time")
        axes[0][3].set_ylabel("steer")
        axes[0][3].set_title("steering")

        # throttle
        axes[1][3].plot(t[: actions.shape[0]], actions[:, 1])
        axes[1][3].set_xlabel("time")
        axes[1][3].set_ylabel("throttle")
        axes[1][3].set_title("throttle")

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes


def plot_spot_state(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    encode_angle: bool = True,
    file_name: str = "state_plot.png",
    step_range: Union[Tuple[int, int], int] = None,
):
    """Plots the state trajectory of the Spot robot"""

    if step_range is not None:
        if isinstance(step_range, int):
            step_range = (0, step_range)
        x = x[step_range[0] : step_range[1]]
        y = y[step_range[0] : step_range[1]]
        u = u[step_range[0] : step_range[1]]

    state_labels = SPOT_STATE_LABELS_ENCODED if encode_angle else SPOT_STATE_LABELS
    state_length = SPOT_STATE_LENGTH_ENCODED if encode_angle else SPOT_STATE_LENGTH
    num_frame_stack = u.shape[1] // SPOT_ACTION_LENGTH
    n_rows, n_cols = 7 if encode_angle else 6, 2
    fig, axs = plt.subplots(n_rows + 3, n_cols, gridspec_kw={'hspace': 0.6, 'wspace': 0.2}, figsize=(32, 32))
    fig.suptitle("Spot State Trajectory Rollout", fontsize=16)

    # y limits
    ang_limit = (-np.pi, np.pi)
    pos_limit = (-4, 4)
    vel_limit = (-3, 3)
    ang_vel_limit = (-2.5, 2.5)
    trig_limit = (-1, 1)
    margin = 0.1

    # map idx to state type, limit and unit
    def get_state_type(idx):
        if encode_angle:
            if idx in [0, 1, 7, 8, 9]:
                return pos_limit, "[m]"
            elif idx in [2, 3]:
                return trig_limit, "[-]"
            elif idx in [4, 5, 10, 11, 12]:
                return vel_limit, "[m/s]"
            elif idx in [6]:
                return ang_vel_limit, "[rad/s]"
        else:
            if idx in [0, 1, 6, 7, 8]:
                return pos_limit, "[m]"
            elif idx in [2]:
                return ang_limit, "[rad]"
            elif idx in [3, 4, 9, 10, 11]:
                return vel_limit, "[m/s]"
            elif idx in [5]:
                return ang_vel_limit, "[rad/s]"
        raise ValueError(f"Invalid idx {idx}")

    def get_action_type(idx):
        if idx in [0, 1, 2, 3, 4, 5]:
            return vel_limit, "[m/s]"
        elif idx in [6]:
            return ang_vel_limit, "[rad/s]"
        raise ValueError(f"Invalid idx {idx}")

    # plot the state
    for idx in range(state_length):
        if encode_angle:
            col_idx = 0 if idx < 7 else 1
            row_idx = idx if idx < 7 else idx - 7
        else:
            col_idx = 0 if idx < 6 else 1
            row_idx = idx if idx < 6 else idx - 6
        ax = axs[row_idx, col_idx]
        ax.plot(x[:, idx], label="true")
        ax.plot(y[:, idx], label="pred", linestyle="--")
        ax.set_title(state_labels[idx])
        y_limit, y_unit = get_state_type(idx)
        ax.set_ylim(y_limit[0] - margin, y_limit[1] + margin)
        ax.set_ylabel(y_unit)
        ax.set_xlabel("Time Step")
        ax.legend()

    # plot the actions
    for idx in range(SPOT_ACTION_LENGTH):
        if encode_angle:
            col_idx = 0 if idx < 3 else 1
            row_idx = idx + 7 if idx < 3 else idx - 3 + 7
        else:
            col_idx = 0 if idx < 3 else 1
            row_idx = idx + 6 if idx < 3 else idx - 3 + 6
        ax = axs[row_idx, col_idx]
        ax.plot(u[:, idx], label="u_t")
        for i in range(1, num_frame_stack):
            ax.plot(u[:, i * SPOT_ACTION_LENGTH + idx], label=f"u_t+{i}")
        ax.set_title(SPOT_ACTION_LABELS[idx])
        y_limit, y_unit = get_action_type(idx)
        ax.set_ylim(y_limit[0] - margin, y_limit[1] + margin)
        ax.set_ylabel(y_unit)
        ax.set_xlabel("Time Step")
        ax.legend()


    if file_name is None:
        return fig
    else:
        dir_name = "results/spot_plots"
        file_name = str(time.time()) + "_" + file_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        path = os.path.join(dir_name, file_name)
        plt.savefig(path)
        plt.close(fig)


def plot_spot_trajectory(
    traj: Union[jnp.array, Transition],
    plot_mode: str,
    num_frame_stack: int = 2,
):
    """Plots the trajectory of the spot robot"""

    # convert transitions to trajectory
    observations = traj.observation
    actions = traj.action
    rewards = traj.reward
    next_observations = traj.next_observation
    print("[util_plotter] Plotting observations with shape", observations.shape)

    # decode angles
    if observations.shape[-1] == SPOT_STATE_LENGTH_ENCODED + SPOT_GOAL_LENGTH:
        observations = decode_angles_spot(observations, angle_idx=SPOT_ANGLE_IDX)
        next_observations = decode_angles_spot(next_observations, angle_idx=SPOT_ANGLE_IDX)

    # define idxs
    goal_dim = SPOT_GOAL_LENGTH
    action_dim = SPOT_ACTION_LENGTH
    state_dim = SPOT_STATE_LENGTH

    if plot_mode == "transitions_eval_full":
        print("[util_plotter] Plotting spot trajectory in transitions_eval_full mode")

        def plot_goal_and_traj(traj_curr, actions_curr):
            fig, axs = plt.subplots(4, 3, figsize=(15, 20))
            # first row: 2D views
            ax1 = axs[0, 0]  # Top view (x vs y)
            ax2 = axs[0, 1]  # Front view (y vs z)
            ax3 = axs[0, 2]  # Side view (x vs z)
    
            # second row: base actions
            ax4 = axs[1, 0]
            ax5 = axs[1, 1]
            ax6 = axs[1, 2]
    
            # third row: ee actions
            ax7 = axs[2, 0]
            ax8 = axs[2, 1]
            ax9 = axs[2, 2]
    
            # fourth row: ee goal distance, ee base distance, rewards
            ax10 = axs[3, 0]
            ax11 = axs[3, 1]
            ax12 = axs[3, 2]

            # 2D view (Top view: x vs y)
            for idx, data in enumerate(traj_curr):
                u = np.cos(data[:, 3])
                v = np.sin(data[:, 3])
                ax1.plot(data[:, 0], data[:, 1], label="Base" if idx == 0 else None)
                ax1.plot(data[:, 6], data[:, 7], label="EE" if idx == 0 else None)
                ax1.quiver(
                    data[:, 0],
                    data[:, 1],
                    u,
                    v,
                    angles="xy",
                    scale_units="xy",
                    scale=3,
                    width=0.001,
                    color="purple",
                    label="Heading" if idx == 0 else None,
                )
                ax1.plot(
                    data[:, state_dim],
                    data[:, state_dim + 1],
                    "ro",
                    label="Goal" if idx == 0 else None,
                )
                ax1.plot(
                    data[0, 0],
                    data[0, 1],
                    "go",
                    label="Base Start" if idx == 0 else None,
                )
                ax1.plot(
                    data[-1, 0],
                    data[-1, 1],
                    "gs",
                    label="Base End" if idx == 0 else None,
                )
                ax1.plot(
                    data[0, 6], data[0, 7], "bo", label="EE Start" if idx == 0 else None
                )
                ax1.plot(
                    data[-1, 6], data[-1, 7], "bs", label="EE End" if idx == 0 else None
                )
            ax1.set_title("2D View of Trajectory (Top)")
            ax1.legend()
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")
            ax1.axis("equal")
            ax1.set_xlim(-2.5, 2.5)
            ax1.set_ylim(-4.5, 4.5)

            # 2D view (Front view: y vs z)
            for idx, data in enumerate(traj_curr):
                ax2.plot(data[:, 1], data[:, 8], label="EE" if idx == 0 else None)
                ax2.plot(
                    data[0, 1], data[0, 8], "ro", label="Start" if idx == 0 else None
                )
                ax2.plot(
                    data[-1, 1], data[-1, 8], "gs", label="End" if idx == 0 else None
                )
                ax2.plot(
                    data[:, state_dim + 1],
                    data[:, state_dim + 2],
                    "r--",
                    label="Goal" if idx == 0 else None,
                )
            ax2.set_title("2D View of Trajectory (Front)")
            ax2.legend()
            ax2.set_xlabel("Y [m]")
            ax2.set_ylabel("Z [m]")
            ax2.axis("equal")
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(0, 1.5)

            # 2D view (Side view: x vs z)
            for idx, data in enumerate(traj_curr):
                ax3.plot(data[:, 0], data[:, 8], label="EE" if idx == 0 else None)
                ax3.plot(
                    data[0, 0], data[0, 8], "ro", label="Start" if idx == 0 else None
                )
                ax3.plot(
                    data[-1, 0], data[-1, 8], "gs", label="End" if idx == 0 else None
                )
                ax3.plot(
                    data[:, state_dim],
                    data[:, state_dim + 2],
                    "r--",
                    label="Goal" if idx == 0 else None,
                )
            ax3.set_title("2D View of Trajectory (Side)")
            ax3.legend()
            ax3.set_xlabel("X [m]")
            ax3.set_ylabel("Z [m]")
            ax3.axis("equal")
            ax3.set_xlim(-2.5, 2.5)
            ax3.set_ylim(0, 1.5)

            # base actions
            for idx, data in enumerate(actions_curr):
                time_steps = np.arange(data.shape[0])
                ax4.plot(time_steps, data[:, 0], label="Base Vx" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 1], label="Base Vy" if idx == 0 else None)
                ax6.plot(
                    time_steps, data[:, 2], label="Base Vtheta" if idx == 0 else None
                )
            ax4.set_title("Base Action Vx")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Vx [m/s]")
            ax4.legend()
            ax4.grid(True)

            ax5.set_title("Base Action Vy")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Vy [m/s]")
            ax5.legend()
            ax5.grid(True)

            ax6.set_title("Base Action Vtheta")
            ax6.set_xlabel("Time Step")
            ax6.set_ylabel("Vtheta [rad/s]")
            ax6.legend()
            ax6.grid(True)

            # ee actions
            for idx, data in enumerate(actions_curr):
                time_steps = np.arange(data.shape[0])
                ax7.plot(time_steps, data[:, 3], label="EE Vx" if idx == 0 else None)
                ax8.plot(time_steps, data[:, 4], label="EE Vy" if idx == 0 else None)
                ax9.plot(time_steps, data[:, 5], label="EE Vz" if idx == 0 else None)
            ax7.set_title("EE Action Vx")
            ax7.set_xlabel("Time Step")
            ax7.set_ylabel("Vx [m/s]")
            ax7.legend()
            ax7.grid(True)

            ax8.set_title("EE Action Vy")
            ax8.set_xlabel("Time Step")
            ax8.set_ylabel("Vy [m/s]")
            ax8.legend()
            ax8.grid(True)

            ax9.set_title("EE Action Vz")
            ax9.set_xlabel("Time Step")
            ax9.set_ylabel("Vz [m/s]")
            ax9.legend()
            ax9.grid(True)

            # ee-goal distance
            for idx, data in enumerate(traj_curr):
                ee_pos = data[:, 6:9]
                goal_pos = data[:, state_dim : state_dim + 3]
                distance = np.linalg.norm(ee_pos - goal_pos, axis=1)
                time_steps = np.arange(data.shape[0])
                ax10.plot(
                    time_steps,
                    distance,
                    label=f"EE-Goal Distance" if idx == 0 else None,
                )
            ax10.set_title("EE-Goal Distance")
            ax10.set_xlabel("Time Step")
            ax10.set_ylabel("Distance [m]")
            ax10.set_ylim(0, 2)
            ax10.legend()
            ax10.grid(True)

            # ee-base distance
            for idx, data in enumerate(traj_curr):
                ee_pos = data[:, 6:9]
                base_pos = data[:, 0:3]
                distance = np.linalg.norm(ee_pos - base_pos, axis=1)
                time_steps = np.arange(data.shape[0])
                ax11.plot(
                    time_steps,
                    distance,
                    label=f"EE-Base Distance" if idx == 0 else None,
                )
            ax11.set_title("EE-Base Distance")
            ax11.set_xlabel("Time Step")
            ax11.set_ylabel("Distance [m]")
            ax11.legend()
            ax11.grid(True)

            # rewards
            for idx, data in enumerate(rewards):
                time_steps = np.arange(data.shape[0])
                ax12.plot(time_steps, data, label="Reward" if idx == 0 else None)
            ax12.set_title("Rewards")
            ax12.set_xlabel("Time Step")
            ax12.set_ylabel("Reward")
            ax12.legend()
            ax12.grid(True)

            plt.tight_layout()
            return fig, axs

        return plot_goal_and_traj(observations, actions)
    elif plot_mode == "transitions_distance_eval":
        print("[util_plotter] Plotting spot trajectory in transitions_distance_eval mode")
        from scipy.spatial.transform import Rotation as R

        def plot_ee_goal_error(trajs):
            # calculate error between EE and Goal
            ee_pos_goal_error = []
            for traj in trajs:
                ee_pos = traj[:, 6:9]
                goal_pos = traj[:, state_dim : state_dim + 3]
                error = np.linalg.norm(ee_pos - goal_pos, axis=1)
                ee_pos_goal_error.append(error)
            ee_pos_goal_error = np.array(ee_pos_goal_error)

            # plot errors
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            fig.subplots_adjust(hspace=0.5)

            # plot all pos errors
            for idx, error in enumerate(ee_pos_goal_error):
                time_steps = np.arange(error.shape[0])
                ax[0].plot(time_steps, error)
            ax[0].axvline(10, color="r", linestyle="--", label="1s - 10 Steps")
            ax[0].set_title("EE-Goal Position Error")
            ax[0].set_xlabel("Time Step")
            ax[0].set_ylabel("Error [m]")
            ax[0].legend()
            ax[0].grid(True)

            # plot mean error and std of pos errors
            mean_error = np.mean(ee_pos_goal_error, axis=0)
            std_error = np.std(ee_pos_goal_error, axis=0)
            time_steps = np.arange(mean_error.shape[0])
            ax[1].plot(time_steps, mean_error, label="Mean Error")
            ax[1].fill_between(
                time_steps,
                mean_error - std_error,
                mean_error + std_error,
                alpha=0.5,
                label="Std Error",
            )
            ax[1].axvline(10, color="r", linestyle="--", label="1s - 10 Steps")
            ax[1].set_title("Mean and Std EE-Goal Position Error")
            ax[1].set_xlabel("Time Step")
            ax[1].set_ylabel("Error [m]")
            ax[1].legend()
            ax[1].grid(True)

            # plot max and min error
            max_error = np.max(ee_pos_goal_error, axis=0)
            min_error = np.min(ee_pos_goal_error, axis=0)
            ax[1].plot(time_steps, max_error, linestyle="--", label="Max Error")
            ax[1].plot(time_steps, min_error, linestyle="--", label="Min Error")
            ax[1].legend()

            # get mean pos error after 10 steps
            mean_pos_error_after_10_steps = np.mean(mean_error[10:])

            return fig, ax, mean_pos_error_after_10_steps

        return plot_ee_goal_error(observations)

    else:
        raise ValueError(f"Invalid plot_mode: {plot_mode}")


def sample_pos_and_goal_spot(
    rng_key: jax.random.PRNGKey,
    domain_lower: jnp.array,
    domain_upper: jnp.array,
    goal_dim: int = SPOT_GOAL_LENGTH,
    state_dim: int = SPOT_STATE_LENGTH,
    standard_init_state: jnp.array = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.92,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,
        ]
    ),
    standard_init_goal: jnp.array = jnp.array(
        [2.0, 0.0, 0.6]
    ),
    max_goal_distance_radius: Optional[float] = 2.0,
    margins: jnp.array = jnp.array(
        [
            0.1,
            0.1,
            0.1 * jnp.pi,
            0.001,
            0.001,
            0.001,
            0.1,
            0.1,
            0.1,
            0.001,
            0.001,
            0.001,
        ]
    ),
) -> Tuple[jnp.array, jnp.array]:
    """Sample a random initial position and goal for the spot robot"""

    # check if the dimensions are correct
    assert (
        len(standard_init_state) == state_dim
    ), f"Invalid state dim {len(standard_init_state)}"
    assert (
        len(standard_init_goal) == goal_dim
    ), f"Invalid goal dim {len(standard_init_goal)}"
    assert (
        len(domain_lower) >= state_dim
    ), f"Invalid domain lower dim {len(domain_lower)}"
    assert (
        len(domain_upper) >= state_dim
    ), f"Invalid domain upper dim {len(domain_upper)}"

    # check if state is within the domain (handle angles separately)
    assert jnp.all(
        (standard_init_state[:2] - margins[:2]) >= domain_lower[:2]
    ), f"State - margins is not within the domain"
    assert jnp.all(
        (standard_init_state[:2] + margins[:2]) <= domain_upper[:2]
    ), f"State + margins is not within the domain"
    assert jnp.all(
        (standard_init_state[3:12] - margins[3:12]) >= domain_lower[3:12]
    ), f"State - margins is not within the domain"
    assert jnp.all(
        (standard_init_state[3:12] + margins[3:12]) <= domain_upper[3:12]
    ), f"State + margins is not within the domain"
    assert jnp.all(
        (project_angle(standard_init_state[2:3] - margins[2])) >= domain_lower[2]
    ), f"Theta - margins is not within the domain"
    assert jnp.all(
        (project_angle(standard_init_state[2:3] + margins[2])) <= domain_upper[2]
    ), f"Theta + margins is not within the domain"

    # check if the goal is within the domain
    if max_goal_distance_radius is None:
        assert jnp.all(
            (standard_init_goal[:3] - margins[6:9]) >= domain_lower[6:9]
        ), f"Goal - margins is not within the domain"
        assert jnp.all(
            (standard_init_goal[:3] + margins[6:9]) <= domain_upper[6:9]
        ), f"Goal + margins is not within the domain"

    # handle keys
    (
        key_goal,
        key_base_pos,
        key_theta,
        key_base_vel,
        key_ee_pos,
        key_ee_vel,
    ) = jax.random.split(rng_key, 6)

    # sample new random initial state
    # base pos
    init_base_pos = standard_init_state[:2] + jax.random.uniform(
        key_base_pos, shape=(2,), minval=-margins[:2], maxval=margins[:2]
    )
    # theta
    init_theta = project_angle(
        standard_init_state[2:3]
        + jax.random.uniform(
            key_theta, shape=(1,), minval=-margins[2], maxval=margins[2]
        )
    )
    # base vel
    init_base_vel = standard_init_state[3:6] + jnp.array(
        margins[3:6]
    ) * jax.random.normal(key_base_vel, shape=(3,))
    # ee pos
    init_ee_pos = standard_init_state[6:9] + jax.random.uniform(
        key_ee_pos, shape=(3,), minval=-margins[6:9], maxval=margins[6:9]
    )
    # ee vel
    init_ee_vel = standard_init_state[9:12] + jnp.array(
        margins[9:12]
    ) * jax.random.normal(key_ee_vel, shape=(3,))

    init_state = jnp.concatenate(
        [
            init_base_pos,
            init_theta,
            init_base_vel,
            init_ee_pos,
            init_ee_vel,
        ]
    )

    assert init_state.shape == (
        state_dim,
    ), f"Invalid init state shape {init_state.shape}"

    # sample new random goal
    if max_goal_distance_radius is not None:

        key_goal_angle, key_goal_distance, key_goal_z = (
            jax.random.split(key_goal, 3)
        )

        # sample goal x,y in front of the robot (direction offset within [-π/2, π/2] relative to theta)
        angle_offset = jax.random.uniform(
            key_goal_angle, minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        total_angle = project_angle(init_theta[0] + angle_offset)
        radius = max_goal_distance_radius * jnp.sqrt(
            jax.random.uniform(key_goal_distance, shape=(), minval=0.0, maxval=1.0)
        )
        x_offset = radius * jnp.cos(total_angle)
        y_offset = radius * jnp.sin(total_angle)
        goal_x = init_ee_pos[0] + x_offset
        goal_y = init_ee_pos[1] + y_offset

        # clip to domain bounds
        goal_x = jnp.clip(goal_x, domain_lower[6], domain_upper[6])
        goal_y = jnp.clip(goal_y, domain_lower[7], domain_upper[7])
        goal_xy = jnp.array([goal_x, goal_y])

        # sample goal z
        goal_z = jax.random.uniform(
            key_goal_z, shape=(), minval=domain_lower[8], maxval=domain_upper[8]
        )

        init_goal = jnp.concatenate([goal_xy, jnp.array([goal_z])])
    else:
        init_goal = standard_init_goal[:3] + jax.random.uniform(
            key_goal, shape=(3,), minval=-margins[6:9], maxval=margins[6:9]
        )
        init_goal = jnp.minimum(
            jnp.maximum(init_goal, domain_lower[6:9]), domain_upper[6:9]
        )

    assert init_goal.shape == (
        goal_dim,
    ), f"Invalid init goal shape {init_goal.shape}"
    return init_state, init_goal


if __name__ == "__main__":
    # test sample_pos_and_goal_spot
    rng_key = jax.random.PRNGKey(0)
    domain_lower = jnp.array(
        [
            # base pos
            -2.5,
            -2.5,
            -jnp.pi,
            # base vel
            -1.6,
            -1.6,
            -1.5,
            # ee pos
            -2.5,
            -2.5,
            0.1,
            # ee vel
            -2.5,
            -2.5,
            -2.5,
            # base action
            -1.6,
            -1.6,
            -1.5,
            # ee action
            -2.5,
            -2.5,
            -2.5,
        ]
    )

    domain_upper = jnp.array(
        [
            # base pos
            4.5,
            2.5,
            jnp.pi,
            # base vel
            1.6,
            1.6,
            1.5,
            # ee pos
            4.5,
            2.5,
            1.8,
            # ee vel
            2.5,
            2.5,
            2.5,
            # base action
            1.6,
            1.6,
            1.5,
            # ee action
            2.5,
            2.5,
            2.5,
        ]
    )
    init_state, init_goal = sample_pos_and_goal_spot(
        rng_key, domain_lower, domain_upper
    )
    print("init_state", init_state)
    print("init_goal", init_goal)
    assert init_state.shape == (12,), f"Invalid init state shape {init_state.shape}"
    assert init_goal.shape == (3,), f"Invalid init goal shape {init_goal.shape}"
    print("Sampled init state and goal successfully")

    # test sample_pos_and_goal_spot with max_goal_distance_radius
    init_state, init_goal = sample_pos_and_goal_spot(
        rng_key, domain_lower, domain_upper, max_goal_distance_radius=2.0
    )
    print("init_state", init_state)
    print("init_goal", init_goal)
    assert init_state.shape == (12,), f"Invalid init state shape {init_state.shape}"
    assert init_goal.shape == (3,), f"Invalid init goal shape {init_goal.shape}"
    print("Sampled init state and goal successfully with max_goal_distance_radius")

    # sample several init states and goals
    n_samples = 10
    init_states = []
    init_goals = []
    keys = jax.random.split(rng_key, n_samples)
    for _ in range(n_samples):
        init_state, init_goal = sample_pos_and_goal_spot(
            keys[_], domain_lower, domain_upper
        )
        init_states.append(init_state)
        init_goals.append(init_goal)
    init_states = jnp.stack(init_states)
    init_goals = jnp.stack(init_goals)
    assert init_states.shape == (
        n_samples,
        12,
    ), f"Invalid init states shape {init_states.shape}"
    assert init_goals.shape == (
        n_samples,
        3,
    ), f"Invalid init goals shape {init_goals.shape}"
    print("Sampled multiple init states and goals successfully")

    # plot sampled init states and goals on xy plane
    fig, axes = plt.subplots(figsize=(8, 8))
    axes.plot(init_states[:, 0], init_states[:, 1], "bo", label="init states")
    axes.plot(init_goals[:, 0], init_goals[:, 1], "ro", label="init goals")
    # add ee position
    axes.plot(init_states[:, 6], init_states[:, 7], "go", label="ee pos")
    # add arrows for heading angle using quiver
    for i in range(n_samples):
        x, y, theta = init_states[i, 0], init_states[i, 1], init_states[i, 2]
        dx, dy = 0.5 * jnp.cos(theta), 0.5 * jnp.sin(theta)
        if i == 0:
            axes.quiver(
                x, y, dx, dy, scale=6, color="black", width=0.005, label="heading"
            )
        else:
            axes.quiver(x, y, dx, dy, scale=6, color="black", width=0.005)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Sampled init states and goals")
    axes.legend()
    fig.savefig("sampled_init_states_goals.png")

    print("All tests passed!")