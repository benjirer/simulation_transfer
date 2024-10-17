import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union
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


def plot_spot_trajectory(
    traj: Union[jnp.array, Transition],
    actions: Optional[jnp.array] = None,
    pos_domain_size: float = 5,
    encode_angle: bool = False,
    plot_mode: str = "simple",
    goal: Optional[jnp.array] = None,
    state_dim: int = 12,
    goal_dim: int = 3,
    action_dim: int = 6,
    num_frame_stack: int = 0,
):
    """Plots the trajectory of the spot robot"""

    # TODO: Fix this according to state dimensions

    assert (state_dim == 12 and not encode_angle) or (
        state_dim == 13 and encode_angle
    ), f"Invalid state dim {state_dim} and encode angle {encode_angle} combination"

    if plot_mode == "extended":
        print(
            "Plotting spot trajectory in extended mode with trajectory shape",
            traj.shape,
        )

        # get state only
        traj = traj[:, :state_dim]

        scale_factor = 1.5
        n_rows = state_dim
        n_rows_extra = 2 if goal is not None else 1
        n_cols = 2 if actions is not None else 1

        if not encode_angle:
            traj = decode_angles(traj, 2)
            state_labels = [
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
        else:
            state_labels = [
                "base_x",
                "base_y",
                "base_theta_sin",
                "base_theta_cos",
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

        fig, axes = plt.subplots(
            nrows=n_rows + n_rows_extra,
            ncols=n_cols,
            figsize=(scale_factor * 8, scale_factor * 10),
        )

        action_labels = [
            "base_vx_action",
            "base_vy_action",
            "base_vtheta_action",
            "ee_vx_action",
            "ee_vy_action",
            "ee_vz_action",
        ]

        traj_min = jnp.min(traj)
        traj_max = jnp.max(traj)
        common_y_range = (traj_min, traj_max)

        for i in range(n_rows):
            axes[i][0].plot(traj[:, i])
            axes[i][0].set_ylabel(state_labels[i])
            axes[i][0].set_title(state_labels[i])
            axes[i][0].set_ylim(common_y_range)

        if actions is not None:
            action_min = jnp.min(actions)
            action_max = jnp.max(actions)
            action_y_range = (action_min, action_max)
            idx_shift = 1 if n_rows == 12 else 2

            axes[3][1].plot(actions[:, 0])
            axes[3][1].set_ylabel(action_labels[0])
            axes[3][1].set_title(action_labels[0])
            axes[3][1].set_ylim(action_y_range)

            axes[4][1].plot(actions[:, 1])
            axes[4][1].set_ylabel(action_labels[1])
            axes[4][1].set_title(action_labels[1])
            axes[4][1].set_ylim(action_y_range)

            axes[5][1].plot(actions[:, 2])
            axes[5][1].set_ylabel(action_labels[2])
            axes[5][1].set_title(action_labels[2])
            axes[5][1].set_ylim(action_y_range)

            axes[9 + idx_shift][1].plot(actions[:, 3])
            axes[9 + idx_shift][1].set_ylabel(action_labels[3])
            axes[9 + idx_shift][1].set_title(action_labels[3])
            axes[9 + idx_shift][1].set_ylim(action_y_range)

            axes[10 + idx_shift][1].plot(actions[:, 4])
            axes[10 + idx_shift][1].set_ylabel(action_labels[4])
            axes[10 + idx_shift][1].set_title(action_labels[4])
            axes[10 + idx_shift][1].set_ylim(action_y_range)

            axes[11 + idx_shift][1].plot(actions[:, 5])
            axes[11 + idx_shift][1].set_xlabel("time")
            axes[11 + idx_shift][1].set_ylabel(action_labels[5])
            axes[11 + idx_shift][1].set_title(action_labels[5])
            axes[11 + idx_shift][1].set_ylim(action_y_range)

            # hide unused action subplots
            axes[0][1].axis("off")
            axes[1][1].axis("off")
            axes[2][1].axis("off")

            axes[6][1].axis("off")
            axes[7][1].axis("off")
            axes[8][1].axis("off")

        # plot distance between base and ee
        base_pos_pre = traj[:, :2]
        base_pos = jnp.concatenate(
            [base_pos_pre, 0.445 * jnp.ones_like(traj[:, :1])], axis=-1
        )
        ee_pos = traj[:, 7:10] if encode_angle else traj[:, 6:9]
        pos_dist = jnp.linalg.norm(base_pos - ee_pos, axis=-1)
        axes[-n_rows_extra][0].plot(pos_dist)
        axes[-n_rows_extra][0].set_ylabel("base_ee_dist")

        if goal is not None:
            assert goal.shape == (3,), f"Invalid goal shape {goal.shape}"
            axes[-1][0].plot(jnp.linalg.norm(ee_pos - goal, axis=-1))
            axes[-1][0].set_ylabel("base_goal_dist")

        axes[-1][0].set_xlabel("time")

        return fig, axes

    elif plot_mode == "simple":
        print(
            "Plotting spot trajectory in simple mode with trajectory shape", traj.shape
        )

        # get state only
        traj = traj[:, :state_dim]

        if encode_angle:
            traj = decode_angles(traj, 2)

        assert traj.shape[-1] == 12, f"Expected 12 states, got {traj.shape[-1]} states"

        state_label_dict = {
            0: "base_x",
            1: "base_y",
            2: "base_theta",
            6: "ee_x",
            7: "ee_y",
            8: "ee_z",
        }

        scale_factor = 1.5
        n_rows = len(state_label_dict)
        n_rows_extra = 1 if goal is not None else 0
        n_cols = 1

        fig, axes = plt.subplots(
            nrows=n_rows + n_rows_extra,
            ncols=n_cols,
            figsize=(scale_factor * 8, scale_factor * 10),
        )
        fig.tight_layout(pad=2.0)

        for i, key in zip(range(n_rows), state_label_dict.keys()):
            if key == 2:
                theta_unwrapped = np.unwrap(traj[:, key].astype(np.float64))
                axes[i].plot(theta_unwrapped)
            else:
                axes[i].plot(traj[:, key])
            axes[i].set_ylabel(state_label_dict[key])
            axes[i].set_title(state_label_dict[key])

        if goal is not None:
            assert goal.shape == (3,), f"Invalid goal shape {goal.shape}"
            ee_pos = traj[:, 6:9]
            axes[-1].plot(jnp.linalg.norm(ee_pos - goal, axis=-1))
            axes[-1].set_ylabel("base_goal_dist")

        axes[-1].set_xlabel("time")

        return fig, axes
    elif plot_mode == "transitions_eval_full":
        print("Plotting spot trajectory in transitions_eval_full mode")

        # convert transitions to trajectory
        observations = traj.observation
        actions = traj.action
        rewards = traj.reward
        next_observations = traj.next_observation

        # decode angles
        if observations.shape[-1] == 16:
            observations = decode_angles(observations, 2)
            next_observations = decode_angles(next_observations, 2)

        def plot_goal_and_traj(traj_curr, actions_curr):
            fig, axs = plt.subplots(1, 5, figsize=(24, 6))
            ax1, ax2, ax3, ax4, ax5 = axs

            # 1. 2D View
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
                    data[:, -3], data[:, -2], "ro", label="Goal" if idx == 0 else None
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

            # 2. Z Component Analysis
            for idx, data in enumerate(traj_curr):
                ee_z = data[:, 8]
                goal_z = data[0, -1]
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

            # 3. EE-Goal Distance
            for idx, data in enumerate(traj_curr):
                ee_pos = data[:, 6:9]
                goal_pos = data[:, -3:]
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

            # 4. EE-Base Distance
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

            # 5. Actions
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
            ax5.set_title("Actions")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Action [m/s or rad/s]")
            ax5.legend()
            ax5.grid(True)

            return fig, axs

        return plot_goal_and_traj(observations, actions)
    elif plot_mode == "transitions_distance_eval":
        print("Plotting spot trajectory in transitions_distance_eval mode")

        # convert transitions to trajectory
        observations = traj.observation
        actions = traj.action
        rewards = traj.reward
        next_observations = traj.next_observation

        # decode angles
        if observations.shape[-1] == 16:
            observations = decode_angles(observations, 2)
            next_observations = decode_angles(next_observations, 2)

        def plot_ee_goal_error(trajs):
            # calculate error between EE and Goal
            ee_goal_error = []
            for traj in trajs:
                ee_pos = traj[:, 6:9]
                goal_pos = traj[:, -3:]
                error = np.linalg.norm(ee_pos - goal_pos, axis=1)
                ee_goal_error.append(error)
            ee_goal_error = np.array(ee_goal_error)

            # plot errors
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            fig.subplots_adjust(hspace=0.5)

            # plot all errors
            for idx, error in enumerate(ee_goal_error):
                time_steps = np.arange(error.shape[0])
                ax[0].plot(time_steps, error)
            ax[0].axvline(10, color="r", linestyle="--", label="1s - 10 Steps")
            ax[0].set_title("EE-Goal Error")
            ax[0].set_xlabel("Time Step")
            ax[0].set_ylabel("Error [m]")
            ax[0].legend()
            ax[0].grid(True)

            # plot mean error and std
            mean_error = np.mean(ee_goal_error, axis=0)
            std_error = np.std(ee_goal_error, axis=0)
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
            ax[1].set_title("Mean and Std EE-Goal Error")
            ax[1].set_xlabel("Time Step")
            ax[1].set_ylabel("Error [m]")
            ax[1].legend()
            ax[1].grid(True)

            # plot max and min error
            max_error = np.max(ee_goal_error, axis=0)
            min_error = np.min(ee_goal_error, axis=0)
            ax[1].plot(time_steps, max_error, linestyle="--", label="Max Error")
            ax[1].plot(time_steps, min_error, linestyle="--", label="Min Error")
            ax[1].legend()

            # get mean error after 10 steps
            mean_error_after_10_steps = np.mean(mean_error[10:])

            return fig, ax, mean_error_after_10_steps

        return plot_ee_goal_error(observations)

    else:
        raise ValueError(f"Invalid plot_mode: {plot_mode}")


def sample_pos_and_goal_spot(
    rng_key: jax.random.PRNGKey,
    domain_lower: jnp.array,
    domain_upper: jnp.array,
    goal_dim: int = 3,
    state_dim: int = 12,
    standard_init_state: jnp.array = jnp.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, -0.012, 0.6, 0.0, 0.0, 0.0]
    ),
    standard_init_goal: jnp.array = jnp.array([2.0, 0.0, 0.6]),
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

    # check if state is within the domain (handle theta separately)
    assert jnp.all(
        (standard_init_state[:2] - margins[:2]) >= domain_lower[:2]
    ), f"State - margins is not within the domain"
    assert jnp.all(
        (standard_init_state[:2] + margins[:2]) <= domain_upper[:2]
    ), f"State + margins is not within the domain"
    assert jnp.all(
        (standard_init_state[3:] - margins[3:]) >= domain_lower[3:12]
    ), f"State - margins is not within the domain"
    assert jnp.all(
        (standard_init_state[3:] + margins[3:]) <= domain_upper[3:12]
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
            (standard_init_goal - margins[6:9]) >= domain_lower[6:9]
        ), f"Goal - margins is not within the domain"
        assert jnp.all(
            (standard_init_goal + margins[6:9]) <= domain_upper[6:9]
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
    init_ee_vel = standard_init_state[9:] + jnp.array(
        margins[9:12]
    ) * jax.random.normal(key_ee_vel, shape=(3,))

    init_state = jnp.concatenate(
        [init_base_pos, init_theta, init_base_vel, init_ee_pos, init_ee_vel]
    )
    assert init_state.shape == (
        state_dim,
    ), f"Invalid init state shape {init_state.shape}"

    # sample new random goal
    if max_goal_distance_radius is not None:

        key_goal_angle, key_goal_distance, key_goal_z = jax.random.split(key_goal, 3)

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
        init_goal = standard_init_goal + jax.random.uniform(
            key_goal, shape=(3,), minval=-margins[6:9], maxval=margins[6:9]
        )
        init_goal = jnp.minimum(
            jnp.maximum(init_goal, domain_lower[6:9]), domain_upper[6:9]
        )

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
