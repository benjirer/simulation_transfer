import time
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp

from sim_transfer.sims.dynamics_models import (
    RaceCar,
    CarParams,
    SpotDynamicsModel,
    SpotParams,
)
from sim_transfer.sims.tolerance_reward import ToleranceReward
from sim_transfer.sims.util import (
    encode_angles,
    decode_angles,
    plot_rc_trajectory,
    plot_spot_trajectory,
)
from sim_transfer.sims.car_sim_config import OBS_NOISE_STD_SIM_CAR
from sim_transfer.sims.spot_sim_config import SPOT_DEFAULT_OBSERVATION_NOISE_STD


class RCCarEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (2,)

    def __init__(
        self,
        goal: jnp.array,
        encode_angle: bool = False,
        ctrl_cost_weight: float = 0.005,
        bound: float = 0.1,
        margin_factor: float = 10.0,
    ):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        # Margin 20 seems to work even better (maybe try at some point)
        self.tolerance_reward = ToleranceReward(
            bounds=(0.0, bound),
            margin=margin_factor * bound,
            value_at_margin=0.1,
            sigmoid="long_tail",
        )

    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """Computes the reward for the given transition"""
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> jnp.array:
        """Computes the reward/penalty for the given action"""
        return -(action**2).sum(-1)

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """Computes the reward for the given observations"""
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist**2 + theta_dist**2)
        reward = self.tolerance_reward(total_dist)
        return reward

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class RCCarSimEnv:
    max_steps: int = 200
    _dt: float = 1 / 30.0
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])
    _init_pose: jnp.array = jnp.array([1.42, -1.04, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = OBS_NOISE_STD_SIM_CAR

    def __init__(
        self,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = False,
        use_obs_noise: bool = True,
        use_tire_model: bool = False,
        action_delay: float = 0.0,
        car_model_params: dict = None,
        margin_factor: float = 10.0,
        max_throttle: float = 1.0,
        car_id: int = 2,
        ctrl_diff_weight: float = 0.0,
        seed: int = 230492394,
        max_steps: int = 200,
    ):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.max_steps = max_steps

        # set car id and corresponding parameters
        assert car_id in [1, 2]
        self.car_id = car_id
        self._set_car_params()

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False)

        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        self._dynamics_params = CarParams(**_car_model_params)
        self._next_step_fn = jax.jit(
            partial(self._dynamics_model.next_step, params=self._dynamics_params)
        )

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = RCCarEnvReward(
            goal=self._goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            margin_factor=margin_factor,
        )

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if abs(action_delay % self._dt) < 1e-8:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array(
                [weight_first, 1.0 - weight_first]
            )
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)
        self.ctrl_diff_weight = ctrl_diff_weight

    def reset(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """Resets the environment to a random initial state close to the initial pose"""
        rng_key = self.rds_key if rng_key is None else rng_key

        # sample random initial state
        key_pos, key_theta, key_vel, key_obs = jax.random.split(rng_key, 4)
        init_pos = self._init_pose[:2] + jax.random.uniform(
            key_pos, shape=(2,), minval=-0.10, maxval=0.10
        )
        init_theta = self._init_pose[2:] + jax.random.uniform(
            key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
        )
        init_vel = jnp.zeros((3,)) + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        self._state = init_state
        self._time = 0
        return self._state_to_obs(self._state, rng_key=key_obs)

    def step(
        self, action: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """Performs one step in the environment

        Args:
            action: array of size (2,) with [steering, throttle]
            rng_key: rng key for the observation noise (optional)
        """

        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])
        # assert jnp.all(-1 <= action) and jnp.all(action <= 1), "action must be in [-1, 1]"
        rng_key = self.rds_key if rng_key is None else rng_key

        jitter_reward = jnp.zeros_like(action).sum(-1)
        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action, jitter_reward = self._get_delayed_action(action)

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state_to_obs(self._state, rng_key=rng_key)

        # compute reward
        reward = (
            self._reward_model.forward(obs=None, action=action, next_obs=obs)
            + jitter_reward
        )

        # check if done
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return (
            obs,
            reward,
            done,
            {"time": self._time, "state": self._state, "reward": reward},
        )

    def _state_to_obs(
        self, state: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        rng_key = self.rds_key if rng_key is None else rng_key

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(
                rng_key, shape=self._state.shape
            )
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def _get_delayed_action(self, action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # push action to action buffer
        last_action = self._action_buffer[-1]
        reward = -self.ctrl_diff_weight * jnp.sum((action - last_action) ** 2)
        self._action_buffer = jnp.concatenate(
            [self._action_buffer[1:], action[None, :]], axis=0
        )

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(
            self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None],
            axis=0,
        )
        assert delayed_action.shape == self.dim_action
        return delayed_action, reward

    @property
    def rds_key(self) -> jax.random.PRNGKey:
        self._rds_key, key = jax.random.split(self._rds_key)
        return key

    @property
    def time(self) -> float:
        return self._time

    def _set_car_params(self):
        from sim_transfer.sims.car_sim_config import (
            DEFAULT_PARAMS_BICYCLE_CAR1,
            DEFAULT_PARAMS_BLEND_CAR1,
            DEFAULT_PARAMS_BICYCLE_CAR2,
            DEFAULT_PARAMS_BLEND_CAR2,
        )

        if self.car_id == 1:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR1
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR1
        elif self.car_id == 2:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR2
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR2
        else:
            raise NotImplementedError(f"Car idx {self.car_id} not supported")


class SpotEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (6,)

    def __init__(
        self,
        goal: jnp.array,
        encode_angle: bool = False,
        ctrl_cost_weight: float = 0.005,
        bound: float = 0.1,
        margin_factor: float = 10.0,
        base_linear_action_cost_weight: float = 1.0,
        base_theta_action_cost_weight: float = 1.0,
        ee_action_cost_weight: float = 1.0,
        only_ee_goal: bool = False,
    ):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.base_linear_action_cost_weight = base_linear_action_cost_weight
        self.base_theta_action_cost_weight = base_theta_action_cost_weight
        if only_ee_goal:
            assert (
                self.goal.shape[-1] == 3
            ), "Goal must be of shape (3,) if only ee goal is True"
        self.only_ee_goal = only_ee_goal
        self.ee_action_cost_weight = ee_action_cost_weight
        self.encode_angle = encode_angle
        self.tolerance_reward = ToleranceReward(
            bounds=(0.0, bound),
            margin=margin_factor * bound,
            value_at_margin=0.1,
            sigmoid="long_tail",
        )

    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """Computes the reward for the given transition"""
        reward_ctrl_base_linear, reward_ctrl_base_theta, reward_ctrl_ee = (
            self.action_reward(action)
        )
        reward_state = self.state_reward(obs, next_obs)
        reward = (
            reward_state
            + self.ctrl_cost_weight
            * self.base_linear_action_cost_weight
            * reward_ctrl_base_linear
            + self.ctrl_cost_weight * self.ee_action_cost_weight * reward_ctrl_ee
            + self.ctrl_cost_weight
            * self.base_theta_action_cost_weight
            * reward_ctrl_base_theta
        )
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """Computes the reward/penalty for the given action"""
        base_linear_action_reward = -(action[:2] ** 2).sum(-1)
        base_theta_action_reward = -(action[2] ** 2)
        ee_action_reward = -(action[3:] ** 2).sum(-1)
        return base_linear_action_reward, base_theta_action_reward, ee_action_reward

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """Computes the reward for the given observations"""
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        if self.only_ee_goal:
            ee_pos_diff = next_obs[..., 6:9] - self.goal[:3]
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(ee_pos_diff), axis=-1))
            total_dist = jnp.sqrt(pos_dist**2)
            reward = self.tolerance_reward(total_dist)
        else:
            base_pos_diff = next_obs[..., :2] - self.goal[:2]
            theta_diff = next_obs[..., 2] - self.goal[2]
            ee_pos_diff = next_obs[..., 6:9] - self.goal[6:9]
            base_pos_dist = jnp.sqrt(jnp.sum(jnp.square(base_pos_diff), axis=-1))
            theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
            ee_pos_dist = jnp.sqrt(jnp.sum(jnp.square(ee_pos_diff), axis=-1))
            total_dist = jnp.sqrt(base_pos_dist**2 + theta_dist**2 + ee_pos_dist**2)
            reward = self.tolerance_reward(total_dist)
        return reward


class SpotSimEnv:
    max_steps: int = 200
    _dt: float = 1 / 30.0
    dim_action: Tuple[int] = (6,)
    _goal: jnp.array = jnp.array(
        [1.0, 1.0, -jnp.pi / 2.0, 0.0, 0.0, 0.0, 1.914, 1.05, 0.6, 0.0, 0.0, 0.0]
    )
    _init_pose: jnp.array = jnp.array(
        [0.0, 0.0, -jnp.pi / 2.0, 0.0, 0.0, 0.0, 0.914, 0.05, 0.7, 0.0, 0.0, 0.0]
    )
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = SPOT_DEFAULT_OBSERVATION_NOISE_STD

    def __init__(
        self,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = False,
        use_obs_noise: bool = True,
        action_delay: float = 0.0,
        spot_model_params: dict = None,
        margin_factor: float = 10.0,
        max_velocity: float = 1.0,
        ctrl_diff_weight: float = 0.0,
        base_linear_action_cost_weight: float = 1.0,
        base_theta_action_cost_weight: float = 1.0,
        ee_action_cost_weight: float = 1.0,
        only_ee_goal: bool = False,
        seed: int = 230492394,
        max_steps: int = 200,
    ):
        """
        Spot robot simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            spot_model_params: dictionary of spot model parameters that overwrite the default values
            margin_factor: factor to scale the margin of the tolerance reward
            max_velocity: maximum velocity of the spot robot
            ctrl_diff_weight: weight of the control difference penalty
            max_steps: maximum number of steps
            seed: random number generator seed
        """

        self.dim_state: Tuple[int] = (13,) if encode_angle else (12,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_velocity = jnp.clip(max_velocity, 0.0, 1.0)
        self.max_steps = max_steps

        # initialize dynamics and observation noise models
        self._dynamics_model = SpotDynamicsModel(dt=self._dt, encode_angle=False)

        # set params
        self._set_spot_params()

        if spot_model_params is None:
            _spot_model_params = self._default_spot_model_params
        else:
            _spot_model_params = self._default_spot_model_params
            _spot_model_params.update(spot_model_params)
        self._dynamics_params = SpotParams(**_spot_model_params)
        self._next_step_fn = jax.jit(
            partial(self._dynamics_model.next_step, params=self._dynamics_params)
        )

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = SpotEnvReward(
            goal=self._goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=encode_angle,
            margin_factor=margin_factor,
            base_linear_action_cost_weight=base_linear_action_cost_weight,
            base_theta_action_cost_weight=base_theta_action_cost_weight,
            ee_action_cost_weight=ee_action_cost_weight,
            only_ee_goal=only_ee_goal,
        )

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if abs(action_delay % self._dt) < 1e-8:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array(
                [weight_first, 1.0 - weight_first]
            )
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)
        self.ctrl_diff_weight = ctrl_diff_weight

    def reset(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """Resets the environment to a random initial state close to the initial pose"""
        rng_key = self.rds_key if rng_key is None else rng_key

        # sample random initial state
        key_base_pos, key_theta, key_base_vel, key_ee_pos, key_ee_vel, key_obs = (
            jax.random.split(rng_key, 6)
        )
        init_base_pos = self._init_pose[:2] + jax.random.uniform(
            key_base_pos, shape=(2,), minval=-0.10, maxval=0.10
        )
        init_theta = self._init_pose[2:3] + jax.random.uniform(
            key_theta, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
        )
        init_base_vel = self._init_pose[3:6] + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_base_vel, shape=(3,))
        init_ee_pos = self._init_pose[6:9] + jax.random.uniform(
            key_ee_pos, shape=(3,), minval=-0.10, maxval=0.10
        )
        init_ee_vel = self._init_pose[9:] + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_ee_vel, shape=(3,))
        init_state = jnp.concatenate(
            [init_base_pos, init_theta, init_base_vel, init_ee_pos, init_ee_vel]
        )

        self._state = init_state
        self._time = 0
        return self._state_to_obs(self._state, rng_key=key_obs)

    def step(
        self, action: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.array, float, bool, Dict[str, Any]]:
        """Performs one step in the environment

        Args:
            action: array of size (6,) with [base_vx, base_vy, base_vw, ee_vx, ee_vy, ee_vz]
            rng_key: rng key for the observation noise (optional)
        """

        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_velocity * action[0])
        action = action.at[1].set(self.max_velocity * action[1])
        action = action.at[2].set(self.max_velocity * action[2])
        action = action.at[3].set(self.max_velocity * action[3])
        action = action.at[4].set(self.max_velocity * action[4])
        action = action.at[5].set(self.max_velocity * action[5])
        rng_key = self.rds_key if rng_key is None else rng_key

        jitter_reward = jnp.zeros_like(action).sum(-1)
        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action, jitter_reward = self._get_delayed_action(action)

        # compute next state
        self._state = self._next_step_fn(self._state, action)
        self._time += 1
        obs = self._state_to_obs(self._state, rng_key=rng_key)

        # compute reward
        reward = (
            self._reward_model.forward(obs=None, action=action, next_obs=obs)
            + jitter_reward
        )

        # check if done
        # TODO: add done if outside of bounds
        done = self._time >= self.max_steps

        # return observation, reward, done, info
        return (
            obs,
            reward,
            done,
            {"time": self._time, "state": self._state, "reward": reward},
        )

    def _state_to_obs(
        self, state: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 12
        rng_key = self.rds_key if rng_key is None else rng_key

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(
                rng_key, shape=self._state.shape
            )
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 13 and self.encode_angle) or (
            obs.shape[-1] == 12 and not self.encode_angle
        )
        return obs

    def _get_delayed_action(self, action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # push action to action buffer
        last_action = self._action_buffer[-1]
        reward = -self.ctrl_diff_weight * jnp.sum((action - last_action) ** 2)
        self._action_buffer = jnp.concatenate(
            [self._action_buffer[1:], action[None, :]], axis=0
        )

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(
            self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None],
            axis=0,
        )
        assert delayed_action.shape == self.dim_action
        return delayed_action, reward

    @property
    def rds_key(self) -> jax.random.PRNGKey:
        self._rds_key, key = jax.random.split(self._rds_key)
        return key

    @property
    def time(self) -> float:
        return self._time

    def _set_spot_params(self):
        from sim_transfer.sims.spot_sim_config import SPOT_DEFAULT_PARAMS

        self._default_spot_model_params: Dict = SPOT_DEFAULT_PARAMS


if __name__ == "__main__":
    # ENCODE_ANGLE = True
    # env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
    #                   action_delay=0.0,
    #                   use_tire_model=True,
    #                   use_obs_noise=True)

    # t_start = time.time()
    # s = env.reset()
    # traj = [s]
    # rewards = []
    # actions = []
    # for i in range(120):
    #     t = i / 30.
    #     a = jnp.array([- 0.5 * jnp.cos(1.0 * t), 0.8 / (t + 1)])
    #     s, r, _, _ = env.step(a)
    #     traj.append(s)
    #     actions.append(a)
    #     rewards.append(r)

    # duration = time.time() - t_start
    # print(f'Duration of trajectory sim {duration} sec')
    # traj = jnp.stack(traj)
    # actions = jnp.stack(actions)

    # plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
    # print(traj[-1, :])

    ENCODE_ANGLE = False
    env = SpotSimEnv(
        encode_angle=ENCODE_ANGLE,
        action_delay=0.0,
        use_obs_noise=True,
        max_velocity=1.5,
    )

    t_start = time.time()
    s = env.reset()
    traj = [s]
    rewards = []
    actions = []
    for i in range(120):
        t = i / 30.0
        a = jnp.array(
            [
                -0.5 * jnp.cos(1.0 * t),
                0.8 / (t + 1),
                0.0,
                0.5 * jnp.sin(1.0 * t),
                0.6 / (t + 1),
                0.0,
            ]
        )
        s, r, _, _ = env.step(a)
        traj.append(s)
        actions.append(a)
        rewards.append(r)

    duration = time.time() - t_start
    print(f"Duration of trajectory sim {duration} sec")
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    # plot trajectory
    plot_spot_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
