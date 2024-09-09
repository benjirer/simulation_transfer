import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


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
    traj: jnp.array,
    actions: Optional[jnp.array] = None,
    pos_domain_size: float = 5,
    encode_angle: bool = False,
    extended: bool = False,
):
    """Plots the trajectory of the spot robot"""

    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    if extended:
        scale_factor = 1.5
        n_rows = traj.shape[-1]
        n_cols = 2 if actions is not None else 1

        if n_rows == 13 and not encode_angle:
            n_rows = 12
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
            nrows=n_rows, ncols=n_cols, figsize=(scale_factor * 8, scale_factor * 10)
        )
        fig.tight_layout(pad=2.0)

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

            if i == n_rows - 1:
                axes[i][0].set_xlabel("time")

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
        return fig, axes
    else:

        if traj.shape[-1] == 13:
            traj = decode_angles(traj, 2)
            assert (
                traj.shape[-1] == 12
            ), f"Expected 12 states, got {traj.shape[-1]} states"

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
        n_cols = 1

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(scale_factor * 8, scale_factor * 10)
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

            if i == n_rows - 1:
                axes[i].set_xlabel("time")

        return fig, axes


def delay_and_stack_spot_actions(
    u: jnp.array,
    action_stacking: bool = False,
    action_delay_base: int = 2,
    action_delay_ee: int = 1,
    action_dim_base: int = 3,
    action_dim_ee: int = 3,
) -> jnp.array:
    """Delay and stack the actions for the spot robot"""
    assert (
        u.shape[-1] == action_dim_base + action_dim_ee
    ), f"Invalid action dimensions, u shape {u.shape[-1]} doesn't match given dims {action_dim_base + action_dim_ee}"
    assert action_delay_base >= 0 and action_delay_ee >= 0, "Action delay must be >= 0"

    # action stacking and action delay
    # note: we can can't split action delay if we use action stacking
    if action_stacking:
        action_delay = max(action_delay_base, action_delay_ee)
        print(
            "Using action stacking with max action delay (we can't split action delay if we use action stacking)",
            action_delay,
        )

        if action_delay > 0:
            u_stacked = jnp.concatenate(
                [
                    jnp.where(
                        jnp.arange(u.shape[0])[:, None] < i,
                        0,
                        jnp.roll(u, shift=-i, axis=0),
                    )
                    for i in range(action_delay + 1)
                ],
                axis=-1,
            )
            assert u_stacked.shape == (
                u.shape[0],
                6 * (action_delay + 1),
            ), "Something went wrong with the action stacking"
            u = u_stacked
    # only action delay
    else:
        u_base, u_ee = u[:, :3], u[:, 3:]
        print(
            "Using action delay for base and ee actions with delays",
            action_delay_base,
            action_delay_ee,
        )

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

    return u
