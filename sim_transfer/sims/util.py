import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, List
import jax.random
import matplotlib.pyplot as plt
from brax.training.types import Transition


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
    x: jnp.ndarray,
    y: jnp.ndarray,
    u: jnp.ndarray,
    encode_angle: bool = True,
    file_name: str = "state_plot.png",
):
    import time
    import os
    from sim_transfer.sims.spot_sim_config import (
        SPOT_STATE_LABELS,
        SPOT_STATE_LABELS_ENCODED,
        SPOT_ACTION_LABELS,
        SPOT_STATE_LENGTH,
        SPOT_ACTION_LENGTH,
        SPOT_GOAL_LENGTH
    )

    state_labels = SPOT_STATE_LABELS_ENCODED if encode_angle else SPOT_STATE_LABELS
    n_rows, n_cols = len(state_labels), 3
    amount_of_actions = u.shape[1] // SPOT_ACTION_LENGTH
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 30))

    for i in range(n_rows):
        for j in range(n_cols - 1):
            axs[i, j].plot(x[:, i], label="x")
            axs[i, j].plot(y[:, i], label="y")
            axs[i, j].set_title(state_labels[i])
            axs[i, j].legend()

    for i in range(len(SPOT_ACTION_LABELS)):
        for j in range(amount_of_actions):

            axs[i, 2].plot(u[:, i + j * 9], label="u")
            axs[i, 2].set_title(SPOT_ACTION_LABELS[i])
            axs[i, 2].legend()

    dir_name = "results/spot_plots"
    file_name = str(time.time()) + "_" + file_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    path = os.path.join(dir_name, file_name)
    plt.savefig(path)


def plot_spot_trajectory(
    traj: Union[jnp.array, Transition],
    plot_mode: str,
    num_frame_stack: int = 0,
):
    """Plots the trajectory of the spot robot"""

    # convert transitions to trajectory
    observations = traj.observation
    actions = traj.action
    rewards = traj.reward
    next_observations = traj.next_observation
    print("Plotting observations with shape", observations.shape)
    assert (
        observations.shape[-1] == 15
        or observations.shape[-1] == 16
        or observations.shape[-1] == 24
        or observations.shape[-1] == 28
    )

    # decode angles
    if observations.shape[-1] == 28:
        # need to decode 2, 12, 13, 14
        observations = decode_angles(observations, 2)
        next_observations = decode_angles(next_observations, 2)
        observations = decode_angles(observations, 12)
        next_observations = decode_angles(next_observations, 12)
        observations = decode_angles(observations, 13)
        next_observations = decode_angles(next_observations, 13)
        observations = decode_angles(observations, 14)
        next_observations = decode_angles(next_observations, 14)

    # define idxs
    goal_dim = 6
    action_dim = 9
    state_dim = 18

    if plot_mode == "transitions_eval_full":
        print("Plotting spot trajectory in transitions_eval_full mode")

        def plot_goal_and_traj(traj_curr, actions_curr):
            fig, axs = plt.subplots(1, 5, figsize=(24, 6))
            ax1, ax2, ax3, ax4, ax5 = axs

            # 2D View
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
            ax1.set_title("2D View of Trajectory")
            ax1.legend()
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")
            ax1.axis("equal")
            ax1.set_xlim(-2.5, 2.5)
            ax1.set_ylim(-4.5, 4.5)

            # Z Component Analysis
            for idx, data in enumerate(traj_curr):
                ee_z = data[:, 8]
                goal_z = data[0, state_dim + 2]
                time_steps = np.arange(data.shape[0])
                ax2.plot(time_steps, ee_z, label=f"EE Z" if idx == 0 else None)
                ax2.hlines(
                    goal_z,
                    time_steps[0],
                    time_steps[-1],
                    colors="r",
                    linestyles="--",
                    label=f"Goal Z" if idx == 0 else None,
                )
                diff_z = ee_z - goal_z
                ax2.plot(
                    time_steps, diff_z, label=f"Difference Z" if idx == 0 else None
                )
            ax2.set_title("EE-Z Position")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Z Position / Difference [m]")
            ax2.legend()
            ax2.grid(True)

            # EE-Pos-Goal Distance
            for idx, data in enumerate(traj_curr):
                ee_pos = data[:, 6:9]
                goal_pos = data[:, state_dim : state_dim + 3]
                distance = np.linalg.norm(ee_pos - goal_pos, axis=1)
                time_steps = np.arange(data.shape[0])
                ax3.plot(
                    time_steps,
                    distance,
                    label=f"EE-Goal Distance" if idx == 0 else None,
                )
            ax3.set_title("EE-Goal Distance")
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel("Distance [m]")
            ax3.set_ylim(0, 2)
            ax3.legend()
            ax3.grid(True)

            # EE-Orient-Goal Distance
            for idx, data in enumerate(traj_curr):
                ee_orient = data[:, 12:15]
                goal_orient = data[:, state_dim + 3 : state_dim + 6]
                # cast both to [-pi, pi]
                ee_orient = (ee_orient + np.pi) % (2 * np.pi) - np.pi
                goal_orient = (goal_orient + np.pi) % (2 * np.pi) - np.pi
                distance = np.linalg.norm(ee_orient - goal_orient, axis=1)
                time_steps = np.arange(data.shape[0])
                ax4.plot(
                    time_steps,
                    distance,
                    label=f"EE-Orient-Goal Distance" if idx == 0 else None,
                )
            ax4.set_title("EE-Orient-Goal Distance")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Distance [rad]")
            ax4.legend()
            ax4.grid(True)

            # EE-Base Distance
            for idx, data in enumerate(traj_curr):
                ee_pos = data[:, 6:9]
                base_pos = data[:, 0:3]
                distance = np.linalg.norm(ee_pos - base_pos, axis=1)
                time_steps = np.arange(data.shape[0])
                ax4.plot(
                    time_steps,
                    distance,
                    label=f"EE-Base Distance" if idx == 0 else None,
                )
            ax4.set_title("EE-Base Distance")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Distance [m]")
            ax4.legend()
            ax4.grid(True)

            # Actions
            for idx, data in enumerate(actions_curr):
                time_steps = np.arange(data.shape[0])
                ax5.plot(time_steps, data[:, 0], label="Base Vx" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 1], label="Base Vy" if idx == 0 else None)
                ax5.plot(
                    time_steps, data[:, 2], label="Base Vtheta" if idx == 0 else None
                )
                ax5.plot(time_steps, data[:, 3], label="EE Vx" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 4], label="EE Vy" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 5], label="EE Vz" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 6], label="EE Roll" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 7], label="EE Pitch" if idx == 0 else None)
                ax5.plot(time_steps, data[:, 8], label="EE Yaw" if idx == 0 else None)
            ax5.set_title("Actions")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Action [m/s or rad/s]")
            ax5.legend()
            ax5.grid(True)

            return fig, axs

        return plot_goal_and_traj(observations, actions)
    elif plot_mode == "transitions_distance_eval":
        print("Plotting spot trajectory in transitions_distance_eval mode")

        def plot_ee_goal_error(trajs):
            # calculate error between EE and Goal
            ee_pos_goal_error = []
            for traj in trajs:
                ee_pos = traj[:, 6:9]
                goal_pos = traj[:, state_dim : state_dim + 3]
                error = np.linalg.norm(ee_pos - goal_pos, axis=1)
                ee_pos_goal_error.append(error)
            ee_pos_goal_error = np.array(ee_pos_goal_error)

            ee_orient_goal_error = []
            for traj in trajs:
                ee_orient = traj[:, 12:15]
                goal_orient = traj[:, state_dim + 3 : state_dim + 6]
                # cast both to [-pi, pi]
                ee_orient = (ee_orient + np.pi) % (2 * np.pi) - np.pi
                goal_orient = (goal_orient + np.pi) % (2 * np.pi) - np.pi
                error = np.linalg.norm(ee_orient - goal_orient, axis=1)
                ee_orient_goal_error.append(error)
            ee_orient_goal_error = np.array(ee_orient_goal_error)

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

            # plot all orient errors
            for idx, error in enumerate(ee_orient_goal_error):
                time_steps = np.arange(error.shape[0])
                ax[2].plot(time_steps, error)
            ax[2].axvline(10, color="r", linestyle="--", label="1s - 10 Steps")
            ax[2].set_title("EE-Goal Orientation Error")
            ax[2].set_xlabel("Time Step")
            ax[2].set_ylabel("Error [rad]")
            ax[2].legend()
            ax[2].grid(True)

            # plot mean error and std of orient errors
            mean_error = np.mean(ee_orient_goal_error, axis=0)
            std_error = np.std(ee_orient_goal_error, axis=0)
            time_steps = np.arange(mean_error.shape[0])
            ax[3].plot(time_steps, mean_error, label="Mean Error")
            ax[3].fill_between(
                time_steps,
                mean_error - std_error,
                mean_error + std_error,
                alpha=0.5,
                label="Std Error",
            )
            ax[3].axvline(10, color="r", linestyle="--", label="1s - 10 Steps")
            ax[3].set_title("Mean and Std EE-Goal Orientation Error")
            ax[3].set_xlabel("Time Step")
            ax[3].set_ylabel("Error [rad]")
            ax[3].legend()
            ax[3].grid(True)

            # plot max and min error
            max_error = np.max(ee_orient_goal_error, axis=0)
            min_error = np.min(ee_orient_goal_error, axis=0)
            ax[3].plot(time_steps, max_error, linestyle="--", label="Max Error")
            ax[3].plot(time_steps, min_error, linestyle="--", label="Min Error")
            ax[3].legend()

            # get mean orient error after 10 steps
            mean_orient_error_after_10_steps = np.mean(mean_error[10:])

            mean_error_after_10_steps = (
                mean_pos_error_after_10_steps,
                mean_orient_error_after_10_steps,
            )
            return fig, ax, mean_error_after_10_steps

        return plot_ee_goal_error(observations)

    else:
        raise ValueError(f"Invalid plot_mode: {plot_mode}")


def sample_pos_and_goal_spot(
    rng_key: jax.random.PRNGKey,
    domain_lower: jnp.array,
    domain_upper: jnp.array,
    goal_dim_with_ee_orientation: int = 6,
    state_dim_with_ee_orientation: int = 18,
    standard_init_state_with_ee_orientation: jnp.array = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.92,
            -0.012,
            0.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    standard_init_goal_with_ee_orientation: jnp.array = jnp.array(
        [2.0, 0.0, 0.6, 0.0, 0.0, 0.0]
    ),
    max_goal_distance_radius: Optional[float] = 2.0,
    margins_with_ee_orientation: jnp.array = jnp.array(
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
            0.1 * jnp.pi,
            0.1 * jnp.pi,
            0.1 * jnp.pi,
            0.001,
            0.001,
            0.001,
        ]
    ),
) -> Tuple[jnp.array, jnp.array]:
    """Sample a random initial position and goal for the spot robot"""

    goal_dim = goal_dim_with_ee_orientation
    state_dim = state_dim_with_ee_orientation
    standard_init_state = standard_init_state_with_ee_orientation
    standard_init_goal = standard_init_goal_with_ee_orientation
    margins = margins_with_ee_orientation

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
    assert jnp.all(
        (project_angle(standard_init_state[12:15] - margins[12:15]))
        >= domain_lower[12:15]
    ), f"EE angles - margins is not within the domain"
    assert jnp.all(
        (project_angle(standard_init_state[12:15] + margins[12:15]))
        <= domain_upper[12:15]
    ), f"EE angles + margins is not within the domain"
    assert jnp.all(
        (standard_init_state[15:18] - margins[15:18]) >= domain_lower[15:18]
    ), f"EE state - margins is not within the domain"
    assert jnp.all(
        (standard_init_state[15:18] + margins[15:18]) <= domain_upper[15:18]
    ), f"EE state + margins is not within the domain"

    # check if the goal is within the domain
    if max_goal_distance_radius is None:
        assert jnp.all(
            (standard_init_goal[:3] - margins[6:9]) >= domain_lower[6:9]
        ), f"Goal - margins is not within the domain"
        assert jnp.all(
            (standard_init_goal[:3] + margins[6:9]) <= domain_upper[6:9]
        ), f"Goal + margins is not within the domain"
        assert jnp.all(
            (project_angle(standard_init_goal[3:] - margins[12:15]))
            >= domain_lower[12:15]
        ), f"Goal - margins is not within the domain"
        assert jnp.all(
            (project_angle(standard_init_goal[3:] + margins[12:15]))
            <= domain_upper[12:15]
        ), f"Goal + margins is not within the domain"

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
    # ee orientation
    init_ee_orientation = project_angle(
        standard_init_state[12:15]
        + jax.random.uniform(
            key_ee_pos, shape=(3,), minval=-margins[12:15], maxval=margins[12:15]
        )
    )
    init_ee_ang_vel = standard_init_state[15:18] + jnp.array(
        margins[15:18]
    ) * jax.random.normal(key_ee_vel, shape=(3,))

    init_state = jnp.concatenate(
        [
            init_base_pos,
            init_theta,
            init_base_vel,
            init_ee_pos,
            init_ee_vel,
            init_ee_orientation,
            init_ee_ang_vel,
        ]
    )

    assert init_state.shape == (
        state_dim,
    ), f"Invalid init state shape {init_state.shape}"

    # sample new random goal
    if max_goal_distance_radius is not None:

        key_goal_angle, key_goal_distance, key_goal_z, key_ee_orientation_goal = (
            jax.random.split(key_goal, 4)
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

        # sample goal ee orientation
        ee_orientation_goal = project_angle(
            standard_init_goal[3:]
            + jax.random.uniform(
                key_ee_orientation_goal,
                shape=(3,),
                minval=-jnp.array([jnp.pi, jnp.pi, jnp.pi]),
                maxval=jnp.array([jnp.pi, jnp.pi, jnp.pi]),
            )
        )
        init_goal = jnp.concatenate([goal_xy, jnp.array([goal_z]), ee_orientation_goal])
    else:
        init_goal = standard_init_goal[:3] + jax.random.uniform(
            key_goal, shape=(3,), minval=-margins[6:9], maxval=margins[6:9]
        )
        init_goal = jnp.minimum(
            jnp.maximum(init_goal, domain_lower[6:9]), domain_upper[6:9]
        )
        init_goal_ee = standard_init_goal[3:] + jax.random.uniform(
            key_goal,
            shape=(3,),
            minval=-jnp.array([jnp.pi, jnp.pi, jnp.pi]),
            maxval=jnp.array([jnp.pi, jnp.pi, jnp.pi]),
        )
        init_goal_ee = jnp.minimum(
            jnp.maximum(init_goal_ee, domain_lower[12:15]), domain_upper[12:15]
        )
        init_goal = jnp.concatenate([init_goal, init_goal_ee])

    return init_state, init_goal


def euler_to_direction_vectorized(rolls, pitches, yaws):
    """Convert arrays of Euler angles to direction vectors."""
    c_r = jnp.cos(rolls)
    s_r = jnp.sin(rolls)
    c_p = jnp.cos(pitches)
    s_p = jnp.sin(pitches)
    c_y = jnp.cos(yaws)
    s_y = jnp.sin(yaws)

    zeros = jnp.zeros_like(rolls)
    ones = jnp.ones_like(rolls)

    Rx = jnp.stack(
        [
            jnp.stack([ones, zeros, zeros], axis=-1),
            jnp.stack([zeros, c_r, -s_r], axis=-1),
            jnp.stack([zeros, s_r, c_r], axis=-1),
        ],
        axis=-2,
    )

    Ry = jnp.stack(
        [
            jnp.stack([c_p, zeros, s_p], axis=-1),
            jnp.stack([zeros, ones, zeros], axis=-1),
            jnp.stack([-s_p, zeros, c_p], axis=-1),
        ],
        axis=-2,
    )

    Rz = jnp.stack(
        [
            jnp.stack([c_y, -s_y, zeros], axis=-1),
            jnp.stack([s_y, c_y, zeros], axis=-1),
            jnp.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    R = jnp.einsum("nij,njk,nkl->nil", Rz, Ry, Rx)
    directions = R[:, :, 0]
    return directions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # test sample_pos_and_goal_spot
    rng_key = jax.random.PRNGKey(0)
    domain_lower = jnp.array(
        [
            # base pos
            -2.5,
            -2.5,
            -jnp.pi,
            # base vel
            -1.0,
            -1.0,
            -1.0,
            # ee pos
            -2.5,
            -2.5,
            0.1,
            # ee vel
            -1.0,
            -1.0,
            -1.0,
            # ee orientation
            -jnp.pi,
            -jnp.pi,
            -jnp.pi,
            # ee ang vel
            -1.0,
            -1.0,
            -1.0,
            # base action
            -1.0,
            -1.0,
            -1.0,
            # ee action
            -1.0,
            -1.0,
            -1.0,
        ]
    )
    domain_upper = jnp.array(
        [
            # base pos
            4.5,
            2.5,
            jnp.pi,
            # base vel
            1.0,
            1.0,
            1.0,
            # ee pos
            4.5,
            2.5,
            1.8,
            # ee vel
            1.0,
            1.0,
            1.0,
            # ee orientation
            jnp.pi,
            jnp.pi,
            jnp.pi,
            # ee ang vel
            1.0,
            1.0,
            1.0,
            # base action
            1.0,
            1.0,
            1.0,
            # ee action
            1.0,
            1.0,
            1.0,
        ]
    )
    init_state, init_goal = sample_pos_and_goal_spot(
        rng_key, domain_lower, domain_upper
    )
    print("init_state", init_state)
    print("init_goal", init_goal)
    assert init_state.shape == (18,), f"Invalid init state shape {init_state.shape}"
    assert init_goal.shape == (6,), f"Invalid init goal shape {init_goal.shape}"
    print("Sampled init state and goal successfully")

    # test sample_pos_and_goal_spot with max_goal_distance_radius
    init_state, init_goal = sample_pos_and_goal_spot(
        rng_key,
        domain_lower,
        domain_upper,
        max_goal_distance_radius=2.0,
    )
    print("init_state", init_state)
    print("init_goal", init_goal)
    assert init_state.shape == (18,), f"Invalid init state shape {init_state.shape}"
    assert init_goal.shape == (6,), f"Invalid init goal shape {init_goal.shape}"
    print("Sampled init state and goal successfully with max_goal_distance_radius")

    # sample several init states and goals
    n_samples = 10
    init_states = []
    init_goals = []
    keys = jax.random.split(rng_key, n_samples)
    for _ in range(n_samples):
        init_state, init_goal = sample_pos_and_goal_spot(
            keys[_],
            domain_lower,
            domain_upper,
        )
        init_states.append(init_state)
        init_goals.append(init_goal)
    init_states = jnp.stack(init_states)
    init_goals = jnp.stack(init_goals)
    assert init_states.shape == (
        n_samples,
        18,
    ), f"Invalid init states shape {init_states.shape}"
    assert init_goals.shape == (
        n_samples,
        6,
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

    # Plot EE positions and orientations in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # EE positions
    ee_positions = np.array(init_states[:, 6:9])
    ax.scatter(
        ee_positions[:, 0],
        ee_positions[:, 1],
        ee_positions[:, 2],
        c="g",
        label="EE positions",
    )

    # EE orientations
    rolls = init_states[:, 12]
    pitches = init_states[:, 13]
    yaws = init_states[:, 14]
    directions = euler_to_direction_vectorized(rolls, pitches, yaws)
    directions = np.array(directions)
    ax.quiver(
        ee_positions[:, 0],
        ee_positions[:, 1],
        ee_positions[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=0.1,
        normalize=True,
        color="g",
    )

    # Goal positions
    goal_positions = np.array(init_goals[:, 0:3])
    ax.scatter(
        goal_positions[:, 0],
        goal_positions[:, 1],
        goal_positions[:, 2],
        c="m",
        label="Goal positions",
    )

    # Goal orientations
    goal_rolls = init_goals[:, 3]
    goal_pitches = init_goals[:, 4]
    goal_yaws = init_goals[:, 5]
    goal_directions = euler_to_direction_vectorized(goal_rolls, goal_pitches, goal_yaws)
    goal_directions = np.array(goal_directions)
    ax.quiver(
        goal_positions[:, 0],
        goal_positions[:, 1],
        goal_positions[:, 2],
        goal_directions[:, 0],
        goal_directions[:, 1],
        goal_directions[:, 2],
        length=0.1,
        normalize=True,
        color="m",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("EE Positions and Orientations")

    # Custom legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="EE positions",
            markerfacecolor="g",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Goal positions",
            markerfacecolor="m",
            markersize=8,
        ),
        Line2D([0], [0], color="g", lw=2, label="EE orientations"),
        Line2D([0], [0], color="m", lw=2, label="Goal orientations"),
    ]
    ax.legend(handles=legend_elements)

    plt.show()
    fig.savefig("ee_positions_orientations.png")

    print("All tests passed!")
