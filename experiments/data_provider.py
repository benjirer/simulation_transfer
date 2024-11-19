import os
import pickle
import glob
import jax
import jax.numpy as jnp
import copy

from functools import partial
from typing import Dict, Any, List, Union, Tuple
from brax.training.types import Transition

from experiments.util import load_csv_recordings
from sim_transfer.sims.car_sim_config import OBS_NOISE_STD_SIM_CAR
from sim_transfer.sims.spot_sim_config import (
    SPOT_DEFAULT_OBSERVATION_NOISE_STD,
    SPOT_DEFAULT_OBSERVATION_NOISE_STD_WITH_EE_ORIENTATION,
)
from sim_transfer.sims.spot_sim_config import (
    SPOT_STATE_MASK,
    SPOT_ACTION_MASK,
    SPOT_GOAL_MASK,
    SPOT_STATE_LENGTH,
    SPOT_STATE_LENGTH_ENCODED,
    SPOT_ACTION_LENGTH,
    SPOT_GOAL_LENGTH,
    SPOT_ANGLE_IDX,
)
from sim_transfer.sims.simulators import StackedActionSimWrapper
from sim_transfer.sims.util import encode_angles as encode_angles_fn
from sim_transfer.sims.util import decode_angles as decode_angles_fn
from sim_transfer.sims.util import encode_angles_spot as encode_angles_spot_fn
from sim_transfer.sims.util import decode_angles_spot as decode_angles_spot_fn
from sim_transfer.sims.util import plot_spot_state


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

DEFAULTS_SINUSOIDS = {
    "obs_noise_std": 0.1,
    "x_support_mode_train": "full",
    "param_mode": "random",
}

DEFAULTS_PENDULUM = {
    "obs_noise_std": 0.02,
    "x_support_mode_train": "full",
    "param_mode": "random",
}

DEFAULTS_GREENHOUSE = {
    "obs_noise_std": 0.05,
    "x_support_mode_train": "full",
    "param_mode": "random",
}

DEFAULTS_SERGIO = {
    "obs_noise_std": 0.05,
    "x_support_mode_train": "full",
    "param_mode": "random",
    "num_cells": 200,
    "num_genes": 10,
}

DEFAULTS_RACECAR = {
    "obs_noise_std": OBS_NOISE_STD_SIM_CAR,
    "x_support_mode_train": "full",
    "param_mode": "random",
}

DEFAULTS_RACECAR_REAL = {"sampling": "consecutive", "num_samples_test": 10000}

DEFAULTS_SPOT = {
    "obs_noise_std": SPOT_DEFAULT_OBSERVATION_NOISE_STD_WITH_EE_ORIENTATION,
    "x_support_mode_train": "full",
    "param_mode": "random",
}

DEFAULTS_SPOT_REAL = {"sampling": "consecutive", "num_samples_test": 1000}

_RACECAR_NOISE_STD_ENCODED = 20 * jnp.concatenate(
    [
        DEFAULTS_RACECAR["obs_noise_std"][:2],
        DEFAULTS_RACECAR["obs_noise_std"][2:3],
        DEFAULTS_RACECAR["obs_noise_std"][2:3],
        DEFAULTS_RACECAR["obs_noise_std"][3:],
    ]
)

_SPOT_NOISE_STD_ENCODED = 20 * jnp.concatenate(
    [
        # TODO Temporary
        DEFAULTS_SPOT["obs_noise_std"][:2],
        DEFAULTS_SPOT["obs_noise_std"][2:3],
        DEFAULTS_SPOT["obs_noise_std"][2:3],
        DEFAULTS_SPOT["obs_noise_std"][3:12],
        DEFAULTS_SPOT["obs_noise_std"][12:13],
        DEFAULTS_SPOT["obs_noise_std"][12:13],
        DEFAULTS_SPOT["obs_noise_std"][13:14],
        DEFAULTS_SPOT["obs_noise_std"][13:14],
        DEFAULTS_SPOT["obs_noise_std"][14:15],
        DEFAULTS_SPOT["obs_noise_std"][14:15],
        DEFAULTS_SPOT["obs_noise_std"][15:],
    ]
)

DATASET_CONFIGS = {
    "sinusoids1d": {
        "likelihood_std": {"value": 0.1},
        "num_samples_train": {"value": 5},
    },
    "sinusoids2d": {
        "likelihood_std": {"value": 0.1},
        "num_samples_train": {"value": 5},
    },
    "pendulum": {
        "likelihood_std": {"value": [0.05, 0.05, 0.5]},
        "num_samples_train": {"value": 20},
    },
    "pendulum_hf": {
        "likelihood_std": {"value": [0.05, 0.05, 0.5]},
        "num_samples_train": {"value": 20},
    },
    "Greenhouse": {
        "likelihood_std": {"value": [0.01 for _ in range(16)]},
        "num_samples_train": {"value": 20},
    },
    "Greenhouse_hf": {
        "likelihood_std": {"value": [0.01 for _ in range(16)]},
        "num_samples_train": {"value": 20},
    },
    "Sergio": {
        "likelihood_std": {
            "value": [0.05 for _ in range(2 * DEFAULTS_SERGIO["num_genes"])]
        },
        "num_samples_train": {"value": 20},
    },
    "Sergio_hf": {
        "likelihood_std": {
            "value": [0.05 for _ in range(2 * DEFAULTS_SERGIO["num_genes"])]
        },
        "num_samples_train": {"value": 20},
    },
    "pendulum_bimodal": {
        "likelihood_std": {"value": [0.05, 0.05, 0.5]},
        "num_samples_train": {"value": 20},
    },
    "racecar": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "racecar_only_pose": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "racecar_no_angvel": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "racecar_hf": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "racecar_hf_only_pose": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "racecar_hf_no_angvel": {
        "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "spot": {
        "likelihood_std": {"value": _SPOT_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 100},
    },
    "spot_real": {
        "likelihood_std": {"value": _SPOT_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 4_400},
    },
    "spot_real_actionstack": {
        "likelihood_std": {"value": _SPOT_NOISE_STD_ENCODED.tolist()},
        "num_samples_train": {"value": 4_400},
    },
}

DATASET_CONFIGS.update(
    {
        name: {
            "likelihood_std": {"value": _RACECAR_NOISE_STD_ENCODED.tolist()},
            "num_samples_train": {"value": 200},
        }
        for name in [
            "real_racecar_new",
            "real_racecar_new_only_pose",
            "real_racecar_new_no_angvel",
            "real_racecar_new_actionstack",
            "real_racecar_v2",
            "real_racecar_v3",
            "real_racecar_v4",
        ]
    }
)


def get_rccar_recorded_data(encode_angle: bool = True, skip_first_n_points: int = 30):
    recordings_dir = os.path.join(DATA_DIR, "recordings_rc_car_v0")
    recording_dfs = load_csv_recordings(recordings_dir)

    def prepare_rccar_data(
        df,
        encode_angles: bool = False,
        change_signs: bool = True,
        skip_first_n: int = 30,
    ):
        u = df[["steer", "throttle"]].to_numpy()
        x = df[["pos x", "pos y", "theta", "s vel x", "s vel y", "s omega"]].to_numpy()
        # project theta into [-\pi, \pi]
        if change_signs:
            x[:, [1, 4]] *= -1
        x[:, 2] = (x[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        if encode_angles:
            x = encode_angles_fn(x, angle_idx=2)
        # remove first n steps (since often not much is happening)
        x, u = x[skip_first_n:], u[skip_first_n:]

        x_data = jnp.concatenate([x[:-1], u[:-1]], axis=-1)  # current state + action
        y_data = x[1:]  # next state
        assert x_data.shape[0] == y_data.shape[0]
        assert x_data.shape[1] - 2 == y_data.shape[1]
        return x_data, y_data

    num_train_traj = 2
    prep_fn = partial(
        prepare_rccar_data, encode_angles=encode_angle, skip_first_n=skip_first_n_points
    )
    x_train, y_train = map(
        lambda x: jnp.concatenate(x, axis=0),
        zip(*map(prep_fn, recording_dfs[:num_train_traj])),
    )
    x_test, y_test = map(
        lambda x: jnp.concatenate(x, axis=0),
        zip(*map(prep_fn, recording_dfs[num_train_traj:])),
    )

    return x_train, y_train, x_test, y_test


def _load_transitions(file_names: List[str]) -> List[Transition]:
    transitions = []
    for fn in file_names:
        with open(fn, "rb") as f:
            data = pickle.load(f)
            assert (
                data.observation.shape[-1] == 12
                and data.next_observation.shape[-1] == 12
            ), "state must be 12D"
            transitions.append(data)
    return transitions


def _rccar_transitions_to_dataset(
    transitions: Transition,
    encode_angles: bool = False,
    skip_first_n: int = 30,
    action_delay: int = 3,
    action_stacking: bool = False,
):
    assert (
        0 <= action_delay <= 3
    ), "Only recorded the last 3 actions and the current action"
    assert action_delay >= 0, "Action delay must be non-negative"

    if action_stacking:
        if action_delay == 0:
            u = transitions.action
        else:
            u = jnp.concatenate(
                [transitions.observation[:, -2 * action_delay :], transitions.action],
                axis=-1,
            )
        assert u.shape[-1] == 2 * (action_delay + 1)
    else:
        if action_delay == 0:
            u = transitions.action
        elif action_delay == 1:
            u = transitions.observation[:, -2:]
        else:
            u = transitions.observation[:, -2 * action_delay : -2 * (action_delay - 1)]
        assert u.shape[-1] == 2

    # make sure that we only take the actual state without the stacked actions
    x = transitions.observation[:, :6]
    y = transitions.next_observation[:, :6]

    # project theta into [-\pi, \pi]
    x[:, 2] = (x[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    y[:, 2] = (y[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    if encode_angles:
        x = encode_angles_fn(x, angle_idx=2)
        y = encode_angles_fn(y, angle_idx=2)

    # remove first n steps (since often not much is happening)
    x, u, y = x[skip_first_n:], u[skip_first_n:], y[skip_first_n:]

    # concatenate state and action
    x_data = jnp.concatenate([x, u], axis=-1)  # current state + action
    y_data = y  # next state

    # check shapes
    assert x_data.shape[0] == y_data.shape[0]
    assert (
        x_data.shape[1] - (2 * (1 + int(action_stacking) * action_delay))
        == y_data.shape[1]
    )

    return x_data, y_data


def get_rccar_recorded_data_new(
    encode_angle: bool = True,
    skip_first_n_points: int = 10,
    dataset: str = "all",
    action_delay: int = 3,
    action_stacking: bool = False,
    car_id: int = 2,
):
    assert car_id in [1, 2, 3]
    if car_id == 1:
        assert dataset in ["all", "v1"]
        recordings_dir = [os.path.join(DATA_DIR, "recordings_rc_car_v1")]
    elif car_id == 2:
        if dataset == "all":
            recordings_dir = [
                os.path.join(DATA_DIR, "recordings_rc_car_v2"),
                os.path.join(DATA_DIR, "recordings_rc_car_v3"),
                os.path.join(DATA_DIR, "recordings_rc_car_v4"),
            ]
            num_test_points = 20_000
        elif dataset in ["v2", "v3", "v4"]:
            recordings_dir = [os.path.join(DATA_DIR, f"recordings_rc_car_{dataset}")]
            num_test_points = 6_000
        else:
            raise ValueError(f"Unknown dataset {dataset} for car_id {car_id}")
    else:
        raise ValueError(f"Unknown car id {car_id}")
    files = [sorted(glob.glob(rd + "/*.pickle")) for rd in recordings_dir]
    file_names = []
    for f in files:
        file_names += f

    # load and shuffle transitions
    transitions = _load_transitions(file_names)

    # transform transitions into supervised learning datasets
    prep_fn = partial(
        _rccar_transitions_to_dataset,
        encode_angles=encode_angle,
        skip_first_n=skip_first_n_points,
        action_delay=action_delay,
        action_stacking=action_stacking,
    )
    x, y = map(lambda x: jnp.concatenate(x, axis=0), zip(*map(prep_fn, transitions)))
    indices = jnp.arange(start=0, stop=x.shape[0], step=1)
    indices = jax.random.shuffle(key=jax.random.PRNGKey(9345), x=indices)
    x, y = x[indices], y[indices]

    # split into train and test
    x_train, y_train, x_test, y_test = (
        x[:-num_test_points],
        y[:-num_test_points],
        x[-num_test_points:],
        y[-num_test_points:],
    )
    return x_train, y_train, x_test, y_test


def _load_spot_datasets(file_path: Union[str, List[str]]):
    if isinstance(file_path, list):
        dataset = []
        for path in file_path:
            with open(path, "rb") as f:
                dataset.append(pickle.load(f))
    else:
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)
    return dataset


def stack_spot_actions(
    u: jnp.array,
    num_stacked_actions: int = 0,
    action_dim: int = SPOT_ACTION_LENGTH,
) -> jnp.array:
    """Stack the actions for the spot robot"""
    assert (
        u.shape[-1] == action_dim
    ), f"Expected u dimension {u.shape[-1]} but got {action_dim}"

    if num_stacked_actions > 0:
        print(
            f"[data_provider] Using action stacking with stacked actions: {num_stacked_actions}",
        )

        # pad to account for the first actions (set to 0)
        u_padded = jnp.pad(u, ((num_stacked_actions, 0), (0, 0)), mode="constant")

        # arrange indices for stacking
        indices = (
            jnp.arange(u.shape[0])[:, None]
            + jnp.arange(-num_stacked_actions, 1)[None, :]
            + num_stacked_actions
        )

        # stack actions
        u_stacked = u_padded[indices].reshape(u.shape[0], -1)
        
        assert u_stacked.shape == (
            u.shape[0],
            action_dim * (num_stacked_actions + 1),
        ), f"Expected stacked u shape {u.shape[0], action_dim * (num_stacked_actions + 1)} but got {u_stacked.shape}"
        u = u_stacked
    return u

from jax.scipy.spatial.transform import Rotation as JaxRotation

def transform_state_omega(states):
    def transform_single_state(s):
        # euler angles: roll (x), pitch (y), yaw (z) of hand frame wrt world frame
        roll = jnp.arctan2(s[13], s[14])   # sin_ee_rx, cos_ee_rx
        pitch = jnp.arctan2(s[15], s[16])  # sin_ee_ry, cos_ee_ry
        yaw = jnp.arctan2(s[17], s[18])    # sin_ee_rz, cos_ee_rz
        rotation = JaxRotation.from_euler('xyz', jnp.array([roll, pitch, yaw]))

        # extract angular velocity (currently in world frame)
        omega_global = s[19:22]

        omega_hand = rotation.apply(omega_global, inverse=True)
        s = s.at[19:22].set(omega_hand)
        return s

    # Vectorize the transformation across the batch
    transformed_states = jax.vmap(transform_single_state)(states)
    return transformed_states

def _prepare_spot_datasets(
    dataset_pre: List[Transition],
    encode_angles: bool = True,
    skip_first_n: int = 30,
    num_stacked_actions: int = 0,
    angle_idx: Union[int, List[int]] = SPOT_ANGLE_IDX,
    add_goal: bool = False,
    action_dim: int = SPOT_ACTION_LENGTH
):
    # unpack dataset
    x = jnp.array([d.observation for d in dataset_pre])
    u = jnp.array([d.action for d in dataset_pre])
    y = jnp.array([d.next_observation for d in dataset_pre])

    # # transform omega from world to hand frame
    # x = transform_state_omega(x)
    # y = transform_state_omega(y)

    assert (
        x.shape[-1] == SPOT_STATE_LENGTH_ENCODED
    ), f"Invalid state dimensions, x shape {x.shape[-1]} doesn't match {SPOT_STATE_LENGTH_ENCODED}"
    assert (
        u.shape[-1] == SPOT_ACTION_LENGTH
    ), f"Invalid action dimensions, u shape {u.shape[-1]} doesn't match {SPOT_ACTION_LENGTH}"
    assert (
        y.shape[-1] == SPOT_STATE_LENGTH_ENCODED
    ), f"Invalid state dimensions, y shape {y.shape[-1]} doesn't match {SPOT_STATE_LENGTH_ENCODED}"

    # plot
    plot_spot_state(x, y, u, encode_angle=encode_angles, file_name="state_plot_raw_new.png")

    # angles are already encoded in the dataset, let's decode them
    x = decode_angles_spot_fn(x, angle_idx=angle_idx)
    y = decode_angles_spot_fn(y, angle_idx=angle_idx)

    # save raw y for goal addition
    y_raw = copy.deepcopy(y)

    # action stacking
    u = stack_spot_actions(
        u=u,
        num_stacked_actions=num_stacked_actions,
    )

    # encode angles
    if encode_angles:
        x = encode_angles_spot_fn(x, angle_idx=angle_idx)
        y = encode_angles_spot_fn(y, angle_idx=angle_idx)

    # remove first n steps (since often not much is happening)
    x, u, y, y_raw = (
        x[skip_first_n:],
        u[skip_first_n:],
        y[skip_first_n:],
        y_raw[skip_first_n:],
    )
    
    # add goal to the state
    if add_goal:
        # steps to look ahead
        k = 10

        # define starting and ending indices for position and orientation
        pos_goal_start_idx = 7 if encode_angles else 6
        pos_goal_end_idx = pos_goal_start_idx + 3
        orient_goal_start_idx = 12
        orient_goal_end_idx = orient_goal_start_idx + 3

        # extract goals
        pos_goal = y[k:, pos_goal_start_idx:pos_goal_end_idx]
        orient_goal = y_raw[k:, orient_goal_start_idx:orient_goal_end_idx]
        # for now: make zeros
        # orient_goal = jnp.zeros_like(orient_goal)
        goal = jnp.concatenate([pos_goal, orient_goal], axis=1)

        # padding for the last goal (repeat the last goal k times)
        last_pos_goal = y[-1, pos_goal_start_idx:pos_goal_end_idx]
        last_orient_goal = y_raw[-1, orient_goal_start_idx:orient_goal_end_idx]
        # for now: make zeros
        # last_orient_goal = jnp.zeros_like(last_orient_goal)
        last_goal = jnp.concatenate([last_pos_goal, last_orient_goal], axis=0)
        padding = jnp.tile(last_goal, (k, 1))
        goal = jnp.concatenate([goal, padding], axis=0)

        assert (
            x.shape[0] == y.shape[0] == u.shape[0] == goal.shape[0]
        ), "Lengths of x, y, u or goal don't match, got: {}, {}, {}, {}".format(
            x.shape, y.shape, u.shape, goal.shape
        )

        # concatenate state and goal
        x = jnp.concatenate([x, goal], axis=-1)
        y = jnp.concatenate([y, goal], axis=-1)

    # create training data
    x_data = jnp.concatenate([x, u], axis=-1)  # current state + action
    y_data = y  # next state

    assert x_data.shape[0] == y_data.shape[0]
    assert x_data.shape[1] - action_dim * (1 + num_stacked_actions) == y_data.shape[1]

    return x_data, y_data


def get_spot_recorded_data(
    encode_angle: bool = True,
    skip_first_n_points: int = 30,
    num_stacked_actions: int = 0,
    num_test_points: int = 1000,
    angle_idx: Union[int, List[int]] = SPOT_ANGLE_IDX,
    add_goal: bool = False,
):
    # load datasets
    recordings_dirs = [
        # os.path.join(DATA_DIR, "recordings_spot_ee_v0"),
        # os.path.join(DATA_DIR, "recordings_spot_ee_v1"),
        # os.path.join(DATA_DIR, "recordings_spot_ee_v2"),
        os.path.join(DATA_DIR, "recordings_spot_ee_v3"),
    ]
    files = sorted(
        [
            file
            for dir_path in recordings_dirs
            for file in glob.glob(os.path.join(dir_path, "*.pickle"))
        ]
    )
    datasets_pre = _load_spot_datasets(files)
    print(f"[data_provider] Loaded {len(datasets_pre)} datasets")

    # transform transitions into supervised learning datasets
    prep_fn = partial(
        _prepare_spot_datasets,
        encode_angles=encode_angle,
        skip_first_n=skip_first_n_points,
        num_stacked_actions=num_stacked_actions,
        angle_idx=angle_idx,
        add_goal=add_goal,
    )
    x, y = map(lambda x: jnp.concatenate(x, axis=0), zip(*map(prep_fn, datasets_pre)))

    # shuffle data
    indices = jnp.arange(start=0, stop=x.shape[0], step=1)
    indices = jax.random.permutation(
        key=jax.random.PRNGKey(9345), x=indices, independent=True
    )
    x, y = x[indices], y[indices]

    # split into train and test
    x_train, y_train, x_test, y_test = (
        x[:-num_test_points],
        y[:-num_test_points],
        x[-num_test_points:],
        y[-num_test_points:],
    )
    return x_train, y_train, x_test, y_test


def provide_data_and_sim(
    data_source: str, data_spec: Dict[str, Any], data_seed: int = 845672
):
    # load data
    key_train, key_test = jax.random.split(jax.random.PRNGKey(data_seed), 2)
    if data_source == "sinusoids1d" or data_source == "sinusoids2d":
        from sim_transfer.sims.simulators import SinusoidsSim

        defaults = DEFAULTS_SINUSOIDS
        sim_hf = sim_lf = SinusoidsSim(input_size=1, output_size=int(data_source[-2]))
        assert (
            {"num_samples_train"}
            <= set(data_spec.keys())
            <= {"num_samples_train"}.union(DEFAULTS_SINUSOIDS.keys())
        )
    elif data_source == "pendulum" or data_source == "pendulum_hf":
        from sim_transfer.sims.simulators import PendulumSim

        defaults = DEFAULTS_PENDULUM
        if data_source == "pendulum_hf":
            sim_hf = PendulumSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumSim(encode_angle=True, high_fidelity=True)
        assert (
            {"num_samples_train"}
            <= set(data_spec.keys())
            <= {"num_samples_train"}.union(DEFAULTS_PENDULUM.keys())
        )
    elif data_source == "Sergio" or data_source == "Sergio_hf":
        defaults = DEFAULTS_SERGIO
        from sim_transfer.sims.simulators import SergioSim

        if data_source == "Sergio_hf":
            sim_hf = SergioSim(
                n_genes=defaults["num_genes"],
                n_cells=defaults["num_cells"],
                use_hf=True,
            )
            sim_lf = SergioSim(
                n_genes=defaults["num_genes"],
                n_cells=defaults["num_cells"],
                use_hf=False,
            )
        else:
            sim_hf = sim_lf = SergioSim(
                n_genes=defaults["num_genes"], n_cells=defaults["n_cells"], use_hf=False
            )
    elif data_source == "Greenhouse" or data_source == "Greenhouse_hf":
        defaults = DEFAULTS_GREENHOUSE
        from sim_transfer.sims.simulators import GreenHouseSim

        if data_source == "Greenhouse_hf":
            sim_hf = GreenHouseSim(use_hf=True)
            sim_lf = GreenHouseSim(use_hf=False)
        else:
            sim_hf = sim_lf = GreenHouseSim(use_hf=False)
    elif data_source == "pendulum_bimodal" or data_source == "pendulum_bimodal_hf":
        from sim_transfer.sims.simulators import PendulumBiModalSim

        defaults = DEFAULTS_PENDULUM
        if data_source == "pendulum_bimodal_hf":
            sim_hf = PendulumBiModalSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumBiModalSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumBiModalSim(encode_angle=True)
        assert (
            {"num_samples_train"}
            <= set(data_spec.keys())
            <= {"num_samples_train"}.union(DEFAULTS_PENDULUM.keys())
        )
    elif data_source.startswith("racecar"):
        from sim_transfer.sims.simulators import RaceCarSim

        defaults = DEFAULTS_RACECAR
        # TODO: Lenart ask Jonas why we always return low fidelity model here:
        if data_source == "racecar_actionstack":
            use_hf_sim = data_spec.get("use_hf_sim", True)
            car_id = data_spec.get("car_id", 2)
            num_stacked_actions = data_spec.get("num_stacked_actions", 3)
            num_test = data_spec.get(
                "num_samples_test", DEFAULTS_RACECAR_REAL["num_samples_test"]
            )

            sim_sample = RaceCarSim(encode_angle=True, use_blend=True, car_id=car_id)
            if num_stacked_actions > 0:
                sim_sample = StackedActionSimWrapper(
                    sim_sample, num_stacked_actions=num_stacked_actions, action_size=2
                )

            # Prepare simulator for bnn_training (the only difference is that here we can have also low fidelity sim)
            sim = RaceCarSim(encode_angle=True, use_blend=use_hf_sim, car_id=car_id)
            if num_stacked_actions > 0:
                sim = StackedActionSimWrapper(
                    sim, num_stacked_actions=num_stacked_actions, action_size=2
                )

            x_train, y_train, x_test, y_test = sim_sample.sample_datasets(
                rng_key=key_train,
                num_samples_train=data_spec["num_samples_train"],
                num_samples_test=num_test,
                obs_noise_std=data_spec.get("obs_noise_std", defaults["obs_noise_std"]),
                x_support_mode_train=data_spec.get(
                    "x_support_mode_train", defaults["x_support_mode_train"]
                ),
                param_mode="typical",
                # Used to be but then we don't sample the right model:
                # param_mode=data_spec.get('param_mode', defaults['param_mode'])
            )

            return x_train, y_train, x_test, y_test, sim
        elif data_source == "racecar_from_true_input_data":
            use_hf_sim = data_spec.get("use_hf_sim", True)
            car_id = data_spec.get("car_id", 2)
            num_stacked_actions = data_spec.get("num_stacked_actions", 3)
            # assert num_stacked_actions == 3, "We only support 3 stacked actions for now"

            # Prepare simulator for bnn_training (the only difference is that here we can have also low fidelity sim)
            sim = RaceCarSim(encode_angle=True, use_blend=use_hf_sim, car_id=car_id)
            if num_stacked_actions > 0:
                sim = StackedActionSimWrapper(
                    sim, num_stacked_actions=num_stacked_actions, action_size=2
                )

            # Now we prepare data
            # 1.st load data from the real car
            x_train, y_train, x_test, y_test = get_rccar_recorded_data_new(
                encode_angle=True,
                action_stacking=True,
                action_delay=num_stacked_actions,
                car_id=car_id,
            )

            # We delete y_train, y_test and replace it with the simulator output
            del y_train, y_test

            num_train_available = x_train.shape[0]
            num_test_available = x_test.shape[0]
            num_train = data_spec["num_samples_train"]
            num_test = data_spec.get(
                "num_samples_test", DEFAULTS_RACECAR_REAL["num_samples_test"]
            )
            assert num_train <= num_train_available and num_test <= num_test_available

            # Subsample input data
            sampling_scheme = data_spec.get(
                "sampling", DEFAULTS_RACECAR_REAL["sampling"]
            )
            if sampling_scheme == "iid":
                # sample random subset (datapoints are not adjacent in time)
                import warnings

                if num_train > num_train_available / 4.0:
                    warnings.warn(
                        f"Not enough data for {num_train} iid samples."
                        f"Requires at lest 4 times as much data as requested "
                        f"iid samples."
                    )
                idx_train = jax.random.choice(
                    key_train,
                    jnp.arange(num_train_available),
                    shape=(num_train,),
                    replace=False,
                )
                idx_test = jax.random.choice(
                    key_test,
                    jnp.arange(num_test_available),
                    shape=(num_test,),
                    replace=False,
                )
            elif sampling_scheme == "consecutive":
                # sample random sub-trajectory (datapoints are adjacent in time -> highly correlated)
                offset_train = jax.random.choice(
                    key_train, jnp.arange(num_train_available - num_train)
                )
                offset_test = jax.random.choice(
                    key_test, jnp.arange(num_test_available - num_test)
                )
                idx_train = jnp.arange(num_train) + offset_train
                idx_test = jnp.arange(num_test) + offset_test
            else:
                raise ValueError(
                    f'Unknown sampling scheme {sampling_scheme}. Needs to be one of ["iid", "consecutive"].'
                )

            x_train, x_test = x_train[idx_train], x_test[idx_test]

            # 2. We obtain next step from the simulator
            sim_for_sampling_data = RaceCarSim(
                encode_angle=True, use_blend=True, car_id=car_id
            )
            if num_stacked_actions > 0:
                sim_for_sampling_data = StackedActionSimWrapper(
                    sim_for_sampling_data,
                    num_stacked_actions=num_stacked_actions,
                    action_size=2,
                )

            y_train = sim_for_sampling_data._typical_f(x_train)
            y_test = sim_for_sampling_data._typical_f(x_test)
            return x_train, y_train, x_test, y_test, sim
        elif data_source == "racecar_hf":
            car_id = data_spec.get("car_id", 2)
            sim_hf = RaceCarSim(
                encode_angle=True, use_blend=True, only_pose=False, car_id=car_id
            )
            sim_lf = RaceCarSim(
                encode_angle=True, use_blend=False, only_pose=False, car_id=car_id
            )
        elif data_source == "racecar_hf_only_pose":
            car_id = data_spec.get("car_id", 2)
            sim_hf = RaceCarSim(
                encode_angle=True, use_blend=True, only_pose=True, car_id=car_id
            )
            sim_lf = RaceCarSim(
                encode_angle=True, use_blend=False, only_pose=True, car_id=car_id
            )
        elif data_source == "racecar_hf_no_angvel":
            car_id = data_spec.get("car_id", 2)
            sim_hf = RaceCarSim(
                encode_angle=True,
                use_blend=True,
                no_angular_velocity=True,
                car_id=car_id,
            )
            sim_lf = RaceCarSim(
                encode_angle=True,
                use_blend=False,
                no_angular_velocity=True,
                car_id=car_id,
            )
        elif data_source == "racecar_only_pose":
            car_id = data_spec.get("car_id", 2)
            sim_hf = sim_lf = RaceCarSim(
                encode_angle=True, use_blend=True, only_pose=True, car_id=car_id
            )
        elif data_source == "racecar_no_angvel":
            car_id = data_spec.get("car_id", 2)
            sim_hf = sim_lf = RaceCarSim(
                encode_angle=True,
                use_blend=True,
                no_angular_velocity=True,
                car_id=car_id,
            )
        elif data_source == "racecar":
            car_id = data_spec.get("car_id", 2)
            sim_hf = sim_lf = RaceCarSim(
                encode_angle=True, use_blend=True, only_pose=False, car_id=car_id
            )
        else:
            raise ValueError(f"Unknown data source {data_source}")
        assert (
            {"num_samples_train"}
            <= set(data_spec.keys())
            <= {"num_samples_train"}.union(DEFAULTS_RACECAR.keys())
        )
    elif data_source.startswith("real_racecar"):
        from sim_transfer.sims.simulators import RaceCarSim

        use_hf_sim = data_spec.get("use_hf_sim", True)
        car_id = data_spec.get("car_id", 2)
        print(
            "[data_provider] Use high-fidelity car sim:", use_hf_sim, "Car id:", car_id
        )

        if data_source.endswith("only_pose"):
            sim_lf = RaceCarSim(
                encode_angle=True, use_blend=use_hf_sim, only_pose=True, car_id=car_id
            )
        elif data_source.endswith("no_angvel"):
            sim_lf = RaceCarSim(
                encode_angle=True,
                use_blend=use_hf_sim,
                no_angular_velocity=True,
                car_id=car_id,
            )
        else:
            sim_lf = RaceCarSim(encode_angle=True, use_blend=use_hf_sim, car_id=car_id)

        if data_source.startswith("real_racecar_new_actionstack"):
            x_train, y_train, x_test, y_test = get_rccar_recorded_data_new(
                encode_angle=True, action_stacking=True, action_delay=3, car_id=car_id
            )
            sim_lf = StackedActionSimWrapper(
                sim_lf, num_stacked_actions=3, action_size=2
            )
        elif data_source.startswith("real_racecar_new"):
            x_train, y_train, x_test, y_test = get_rccar_recorded_data_new(
                encode_angle=True, action_stacking=False, action_delay=3, car_id=car_id
            )
        elif data_source.startswith("real_racecar_v3"):
            x_train, y_train, x_test, y_test = get_rccar_recorded_data_new(
                encode_angle=True,
                action_stacking=False,
                action_delay=3,
                car_id=car_id,
                dataset="v3",
            )
        else:
            x_train, y_train, x_test, y_test = get_rccar_recorded_data(
                encode_angle=True
            )

        num_train_available = x_train.shape[0]
        num_test_available = x_test.shape[0]

        num_train = data_spec["num_samples_train"]
        num_test = data_spec.get(
            "num_samples_test", DEFAULTS_RACECAR_REAL["num_samples_test"]
        )
        assert num_train <= num_train_available and num_test <= num_test_available
        sampling_scheme = data_spec.get("sampling", DEFAULTS_RACECAR_REAL["sampling"])
        if sampling_scheme == "iid":
            # sample random subset (datapoints are not adjacent in time)
            import warnings

            if num_train > num_train_available / 4.0:
                warnings.warn(
                    f"Not enough data for {num_train} iid samples."
                    f"Requires at lest 4 times as much data as requested "
                    f"iid samples."
                )
            idx_train = jax.random.choice(
                key_train,
                jnp.arange(num_train_available),
                shape=(num_train,),
                replace=False,
            )
            idx_test = jax.random.choice(
                key_test,
                jnp.arange(num_test_available),
                shape=(num_test,),
                replace=False,
            )
        elif sampling_scheme == "consecutive":
            # sample random sub-trajectory (datapoints are adjacent in time -> highly correlated)
            offset_train = jax.random.choice(
                key_train, jnp.arange(num_train_available - num_train)
            )
            offset_test = jax.random.choice(
                key_test, jnp.arange(num_test_available - num_test + 1)
            )
            idx_train = jnp.arange(num_train) + offset_train
            idx_test = jnp.arange(num_test) + offset_test
        else:
            raise ValueError(
                f'Unknown sampling scheme {sampling_scheme}. Needs to be one of ["iid", "consecutive"].'
            )

        x_train, y_train, x_test, y_test = (
            x_train[idx_train],
            y_train[idx_train],
            x_test[idx_test],
            y_test[idx_test],
        )
        if data_source.endswith("only_pose"):
            y_train = y_train[..., :-3]
            y_test = y_test[..., :-3]
        elif data_source.endswith("no_angvel"):
            y_train = y_train[..., :-1]
            y_test = y_test[..., :-1]
        return x_train, y_train, x_test, y_test, sim_lf

    # Spot robot
    elif data_source.startswith("spot"):
        from sim_transfer.sims.simulators import SpotSim

        defaults = DEFAULTS_SPOT
        sampling_scheme = data_spec.get("sampling", DEFAULTS_SPOT_REAL["sampling"])

        # get real recorded data without goal
        if data_source == "spot_real":
            num_stacked_actions = data_spec.get("num_stacked_actions", 0)
            if num_stacked_actions > 0:
                sim = StackedActionSimWrapper(
                    SpotSim(encode_angle=True),
                    num_stacked_actions=num_stacked_actions,
                    action_size=SPOT_ACTION_LENGTH,
                )
            else:
                sim = SpotSim(encode_angle=True)
            print(
                f"[data_provider] Using real Spot data with {num_stacked_actions} stacked actions"
            )
            x_train, y_train, x_test, y_test = get_spot_recorded_data(
                encode_angle=True,
                add_goal=False,
                num_stacked_actions=num_stacked_actions,
            )

        # get real recorded data with goal
        elif data_source == "spot_real_with_goal":
            num_stacked_actions = data_spec.get("num_stacked_actions", 0)
            if num_stacked_actions > 0:
                sim = StackedActionSimWrapper(
                    SpotSim(encode_angle=True),
                    num_stacked_actions=num_stacked_actions,
                    action_size=SPOT_ACTION_LENGTH,
                )
            else:
                sim = SpotSim(encode_angle=True)
            print(
                f"[data_provider] Using real Spot data with goal and {num_stacked_actions} stacked actions"
            )
            x_train, y_train, x_test, y_test = get_spot_recorded_data(
                encode_angle=True,
                add_goal=True,
                num_stacked_actions=num_stacked_actions,
            )
        else:
            raise ValueError(f"Unknown spot data source {data_source}")

        # sample data
        num_train_available = x_train.shape[0]
        num_test_available = x_test.shape[0]
        num_train = data_spec["num_samples_train"]
        num_test = data_spec.get(
            "num_samples_test", DEFAULTS_SPOT_REAL["num_samples_test"]
        )
        assert (
            num_train <= num_train_available and num_test <= num_test_available
        ), f"{num_train} > {num_train_available} or {num_test} > {num_test_available}"
        if sampling_scheme == "iid":
            # sample random subset (datapoints are not adjacent in time)
            import warnings

            if num_train > num_train_available / 4.0:
                warnings.warn(
                    f"Not enough data for {num_train} iid samples."
                    f"Requires at lest 4 times as much data as requested "
                    f"iid samples."
                )
            idx_train = jax.random.choice(
                key_train,
                jnp.arange(num_train_available),
                shape=(num_train,),
                replace=False,
            )
            idx_test = jax.random.choice(
                key_test,
                jnp.arange(num_test_available),
                shape=(num_test,),
                replace=False,
            )
        elif sampling_scheme == "consecutive":
            # sample random sub-trajectory (datapoints are adjacent in time -> highly correlated)
            offset_train = jax.random.choice(
                key_train, jnp.arange(num_train_available - num_train)
            )
            offset_test = jax.random.choice(
                key_test, jnp.arange(num_test_available - num_test + 1)
            )
            idx_train = jnp.arange(num_train) + offset_train
            idx_test = jnp.arange(num_test) + offset_test
        else:
            raise ValueError(
                f'Unknown sampling scheme {sampling_scheme}. Needs to be one of ["iid", "consecutive"].'
            )

        x_train, y_train, x_test, y_test = (
            x_train[idx_train],
            y_train[idx_train],
            x_test[idx_test],
            y_test[idx_test],
        )
        return x_train, y_train, x_test, y_test, sim

    else:
        raise ValueError("Unknown data source %s" % data_source)

    x_train, y_train, x_test, y_test = sim_hf.sample_datasets(
        rng_key=key_train,
        num_samples_train=data_spec["num_samples_train"],
        num_samples_test=1000,
        obs_noise_std=data_spec.get("obs_noise_std", defaults["obs_noise_std"]),
        x_support_mode_train=data_spec.get(
            "x_support_mode_train", defaults["x_support_mode_train"]
        ),
        param_mode=data_spec.get("param_mode", defaults["param_mode"]),
    )
    return x_train, y_train, x_test, y_test, sim_lf


# sample data directly from spot simulator
def sample_from_spot_sim(
    num_samples: int = 1000,
    num_stacked_actions: int = 0,
    eval_directly: bool = False,
):
    from sim_transfer.sims.simulators import SpotSim

    sim = SpotSim(encode_angle=True)
    if num_stacked_actions > 0:
        sim = StackedActionSimWrapper(
            sim, num_stacked_actions=num_stacked_actions, action_size=SPOT_ACTION_LENGTH
        )

    if not eval_directly:
        # define actions u
        # u = base_vx, base_vy, base_vtheta, ee_vx, ee_vy, ee_vz, ee_vrx, ee_vry, ee_vrz
        # u has shape (num_samples, 9)
        # make actions be sin of time for vrx, zero for all others
        t = jnp.linspace(0, 10, num_samples)
        u = jnp.zeros((num_samples, 9))
        u = u.at[:, 6].set(jnp.sin(t))
        u = u.at[:, 7].set(jnp.cos(t))

        # u 0 sould be 0.1
        u = u.at[:, 0].set(0.1)
        u = u.at[:, 4].set(0.1)

        # stack the actions
        u = stack_spot_actions(
            u=u,
            num_stacked_actions=num_stacked_actions,
            action_dim=SPOT_ACTION_LENGTH,
        )

        # define initial state x
        x_init = jnp.zeros((SPOT_STATE_LENGTH))
        x_init = jnp.expand_dims(x_init, axis=0)

        x_train = []
        y_train = []

        # simulate the system for num_samples
        x = x_init
        for i in range(num_samples):
            u_input = jnp.expand_dims(u[i], axis=0)
            x_input = jnp.concatenate([x, u_input], axis=-1)
            y = sim._typical_f(x_input)
            x_train.append(x)
            y_train.append(y)
            x = y

        x_train = jnp.concatenate(x_train, axis=0)
        x_train = jnp.concatenate([x_train, u], axis=-1)
        y_train = jnp.concatenate(y_train, axis=0)

        # split
        amount_train = int(num_samples * 0.8)
        x_train, y_train = x_train[:amount_train], y_train[:amount_train]
        x_test, y_test = x_train[amount_train:], y_train[amount_train:]

    else:
        x_train, y_train, x_test, y_test = sim.sample_datasets(
            rng_key=jax.random.PRNGKey(1234),
            num_samples_train=int(num_samples * 0.8),
            num_samples_test=int(num_samples * 0.2),
            obs_noise_std=SPOT_DEFAULT_OBSERVATION_NOISE_STD_WITH_EE_ORIENTATION,
            param_mode="typical",
        )

    return x_train, y_train, x_test, y_test, sim


if __name__ == "__main__":
    # x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='real_racecar_new',
    #                                                              data_spec={'num_samples_train': 10000})
    # x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='racecar_hf',
    #                                                              data_spec={'num_samples_train': 10000})

    # test spot data
    # x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
    #     data_source="spot_real",
    #     data_spec={"num_samples_train": 5000, "num_samples_test": 50, "num_stacked_actions": 2},
    # )
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # print(jnp.max(x_train, axis=0))

    # test spot sim data sampinling
    num_stacked_actions = 2
    x, y, x_test, y_test, sim = sample_from_spot_sim(
        num_samples=100, num_stacked_actions=num_stacked_actions, eval_directly=True
    )
    # plot all data
    import matplotlib.pyplot as plt

    # labels
    state_labels = [
        "base_x",
        "base_y",
        "sin_base_theta",
        "cos_base_theta",
        "base_vx",
        "base_vy",
        "base_vtheta",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_vx",
        "ee_vy",
        "ee_vz",
        "ee_rx",
        "ee_ry",
        "ee_rz",
        "ee_vrx",
        "ee_vry",
        "ee_vrz",
    ]

    action_labels = [
        "base_vx",
        "base_vy",
        "base_vtheta",
        "ee_vx",
        "ee_vy",
        "ee_vz",
        "ee_vrx",
        "ee_vry",
        "ee_vrz",
    ]

    n_rows = len(state_labels) + len(action_labels)
    n_cols = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 30))

    for i in range(n_rows):
        if i < len(state_labels):
            # axs[i, 0].plot(x[:, i], label='x')
            axs[i, 0].plot(y[:, i], label="y")
            axs[i, 0].set_title(f"{state_labels[i]}")
        else:
            axs[i, 0].plot(x[:, i], label="u_t_0")
            if num_stacked_actions > 0:
                axs[i, 0].plot(x[:, i + 9], label="u_t_1")
            if num_stacked_actions > 1:
                axs[i, 0].plot(x[:, i + 18], label="u_t_2")
            axs[i, 0].set_title(f"{action_labels[i - len(state_labels)]}")
        axs[i, 0].legend()

    # save in file
    plt.savefig("sim_sampled_data.png")
