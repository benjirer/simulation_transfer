from typing import Dict, Any
from functools import partial
import jax.numpy as jnp
import jax
import os

from sim_transfer.sims.util import encode_angles as encode_angles_fn
from experiments.util import load_csv_recordings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


DEFAULTS_SINUSOIDS = {
    'obs_noise_std': 0.1,
    'x_support_mode_train': 'full',
    'param_mode': 'random',
}

DEFAULTS_PENDULUM = {
    'obs_noise_std': 0.02,
    'x_support_mode_train': 'full',
    'param_mode': 'random'
}

DEFAULTS_RACECAR = {
    'obs_noise_std': 0.05 * jnp.exp(jnp.array([-3.3170326, -3.7336411, -2.7081904,
                                               -2.7841284, -2.7067015, -1.4446207])),
    'x_support_mode_train': 'full',
    'param_mode': 'random'
}

DEFAULTS_RACECAR_REAL = {
    'sampling': 'consecutive',
    'num_samples_test': 4000
}

_RACECAR_NOISE_STD_ENCODED = 40 * jnp.concatenate([DEFAULTS_RACECAR['obs_noise_std'][:2],
                                                  DEFAULTS_RACECAR['obs_noise_std'][2:3],
                                                  DEFAULTS_RACECAR['obs_noise_std'][2:3],
                                                  DEFAULTS_RACECAR['obs_noise_std'][3:]])

DATASET_CONFIGS = {
    'sinusoids1d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'sinusoids2d': {
        'likelihood_std': {'value': 0.1},
        'num_samples_train': {'value': 5},
    },
    'pendulum': {
        'likelihood_std': {'value': [0.05, 0.05, 0.5]},
        'num_samples_train': {'value': 20},
    },
    'pendulum_hf': {
        'likelihood_std': {'value': [0.05, 0.05, 0.5]},
        'num_samples_train': {'value': 20},
    },
    'pendulum_bimodal': {
        'likelihood_std': {'value': [0.05, 0.05, 0.5]},
        'num_samples_train': {'value': 20},
    },
    'racecar': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'racecar_only_pose': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'racecar_no_angvel': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'racecar_hf': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'racecar_hf_only_pose': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'racecar_hf_no_angvel': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 100},
    },
    'real_racecar': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 200},
    },
    'real_racecar_only_pose': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 200},
    },
    'real_racecar_no_angvel': {
        'likelihood_std': {'value': _RACECAR_NOISE_STD_ENCODED.tolist()},
        'num_samples_train': {'value': 200},
    },

}


def get_rccar_recorded_data(encode_angle: bool = True, skip_first_n_points: int = 30):
    recordings_dir = os.path.join(DATA_DIR, 'recordings_rc_car_v0')
    recording_dfs = load_csv_recordings(recordings_dir)

    def prepare_rccar_data(df, encode_angles: bool = False, change_signs: bool = True,
                           skip_first_n: int = 30):
        u = df[['steer', 'throttle']].to_numpy()
        x = df[['pos x', 'pos y', 'theta', 's vel x', 's vel y', 's omega']].to_numpy()
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

    num_train_traj = 1
    prep_fn = partial(prepare_rccar_data, encode_angles=encode_angle, skip_first_n=skip_first_n_points)
    x_train, y_train = map(lambda x: jnp.concatenate(x, axis=0), zip(*map(prep_fn, recording_dfs[:num_train_traj])))
    x_test, y_test = map(lambda x: jnp.concatenate(x, axis=0), zip(*map(prep_fn, recording_dfs[num_train_traj:])))

    return x_train, y_train, x_test, y_test


def provide_data_and_sim(data_source: str, data_spec: Dict[str, Any], data_seed: int = 845672):
    # load data
    key_train, key_test = jax.random.split(jax.random.PRNGKey(data_seed), 2)
    if data_source == 'sinusoids1d' or data_source == 'sinusoids2d':
        from sim_transfer.sims.simulators import SinusoidsSim
        defaults = DEFAULTS_SINUSOIDS
        sim_hf = sim_lf = SinusoidsSim(input_size=1, output_size=int(data_source[-2]))
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_SINUSOIDS.keys())
    elif data_source == 'pendulum' or data_source == 'pendulum_hf':
        from sim_transfer.sims.simulators import PendulumSim
        defaults = DEFAULTS_PENDULUM
        if data_source == 'pendulum_hf':
            sim_hf = PendulumSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumSim(encode_angle=True, high_fidelity=False)
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_PENDULUM.keys())
    elif data_source == 'pendulum_bimodal' or data_source == 'pendulum_bimodal_hf':
        from sim_transfer.sims.simulators import PendulumBiModalSim
        defaults = DEFAULTS_PENDULUM
        if data_source == 'pendulum_bimodal_hf':
            sim_hf = PendulumBiModalSim(encode_angle=True, high_fidelity=True)
            sim_lf = PendulumBiModalSim(encode_angle=True, high_fidelity=False)
        else:
            sim_hf = sim_lf = PendulumBiModalSim(encode_angle=True)
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_PENDULUM.keys())
    elif data_source.startswith('racecar'):
        from sim_transfer.sims.simulators import RaceCarSim
        defaults = DEFAULTS_RACECAR
        if data_source == 'racecar_hf':
            sim_hf = RaceCarSim(encode_angle=True, use_blend=True, only_pose=False)
            sim_lf = RaceCarSim(encode_angle=True, use_blend=False, only_pose=False)
        elif data_source == 'racecar_hf_only_pose':
            sim_hf = RaceCarSim(encode_angle=True, use_blend=True, only_pose=True)
            sim_lf = RaceCarSim(encode_angle=True, use_blend=False, only_pose=True)
        elif data_source == 'racecar_hf_no_angvel':
            sim_hf = RaceCarSim(encode_angle=True, use_blend=True, no_angular_velocity=True)
            sim_lf = RaceCarSim(encode_angle=True, use_blend=False, no_angular_velocity=True)
        elif data_source == 'racecar_only_pose':
            sim_hf = sim_lf = RaceCarSim(encode_angle=True, use_blend=True, only_pose=True)
        elif data_source == 'racecar_no_angvel':
            sim_hf = sim_lf = RaceCarSim(encode_angle=True, use_blend=True, no_angular_velocity=True)
        elif data_source == 'racecar':
            sim_hf = sim_lf = RaceCarSim(encode_angle=True, use_blend=True, only_pose=False)
        else:
            raise ValueError(f'Unknown data source {data_source}')
        assert {'num_samples_train'} <= set(data_spec.keys()) <= {'num_samples_train'}.union(DEFAULTS_RACECAR.keys())
    elif data_source.startswith('real_racecar'):
        from sim_transfer.sims.simulators import RaceCarSim

        if data_source.endswith('only_pose'):
            sim_lf = RaceCarSim(encode_angle=True, use_blend=True, only_pose=True)
        elif data_source.endswith('no_angvel'):
            sim_lf = RaceCarSim(encode_angle=True, use_blend=True, no_angular_velocity=True)
        else:
            sim_lf = RaceCarSim(encode_angle=True, use_blend=True)

        x_train, y_train, x_test, y_test = get_rccar_recorded_data(encode_angle=True)
        num_train_available = x_train.shape[0]
        num_test_available = x_test.shape[0]

        num_train = data_spec['num_samples_train']
        num_test = data_spec.get('num_samples_test', DEFAULTS_RACECAR_REAL['num_samples_test'])
        assert num_train <= num_train_available and num_test <= num_test_available
        sampling_scheme = data_spec.get('sampling', DEFAULTS_RACECAR_REAL['sampling'])
        if sampling_scheme == 'iid':
            # sample random subset (datapoints are not adjacent in time)
            assert num_train <= num_train_available / 4., f'Not enough data for {num_train} iid samples.' \
                                                f'Requires at lest 4 times as much data as requested iid samples.'
            idx_train = jax.random.choice(key_train, jnp.arange(num_train_available), shape=(num_train,), replace=False)
            idx_test = jax.random.choice(key_test, jnp.arange(num_test_available), shape=(num_test,), replace=False)
        elif sampling_scheme == 'consecutive':
            # sample random sub-trajectory (datapoints are adjacent in time -> highly correlated)
            offset_train = jax.random.choice(key_train, jnp.arange(num_train_available - num_train))
            offset_test = jax.random.choice(key_test, jnp.arange(num_test_available - num_test))
            idx_train = jnp.arange(num_train) + offset_train
            idx_test = jnp.arange(num_test) + offset_test
        else:
            raise ValueError(f'Unknown sampling scheme {sampling_scheme}. Needs to be one of ["iid", "consecutive"].')

        x_train, y_train, x_test, y_test = x_train[idx_train], y_train[idx_train], x_test[idx_test], y_test[idx_test]
        if data_source.endswith('only_pose'):
            y_train = y_train[..., :-3]
            y_test = y_test[..., :-3]
        elif data_source.endswith('no_angvel'):
            y_train = y_train[..., :-1]
            y_test = y_test[..., :-1]
        return x_train, y_train, x_test, y_test, sim_lf

    else:
        raise ValueError('Unknown data source %s' % data_source)

    x_train, y_train, x_test, y_test = sim_hf.sample_datasets(
        rng_key=key_train,
        num_samples_train=data_spec['num_samples_train'],
        num_samples_test=1000,
        obs_noise_std=data_spec.get('obs_noise_std', defaults['obs_noise_std']),
        x_support_mode_train=data_spec.get('x_support_mode_train', defaults['x_support_mode_train']),
        param_mode=data_spec.get('param_mode', defaults['param_mode'])
    )
    return x_train, y_train, x_test, y_test, sim_lf
