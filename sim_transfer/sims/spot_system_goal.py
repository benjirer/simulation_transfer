import time
from functools import partial
from typing import Dict, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution
from distrax import Normal
from jax import jit, debug
from mbpo.systems.base_systems import (
    Dynamics,
    Reward,
    System,
    SystemState,
    SystemParams,
    DynamicsParams,
    RewardParams,
)

from sim_transfer.sims.dynamics_models import SpotDynamicsModel, SpotParams
from sim_transfer.sims.envs import SpotEnvReward
from sim_transfer.sims.util import (
    encode_angles,
    plot_spot_trajectory,
    sample_pos_and_goal_spot,
)
from sim_transfer.sims.simulators import SpotSim


@chex.dataclass
class SpotDynamicsParams:
    action_buffer: chex.Array
    spot_params: SpotParams
    key: chex.PRNGKey


class SpotDynamics(Dynamics[SpotDynamicsParams]):
    max_steps: int = 200
    _dt: float = 1 / 10.0
    dim_action: Tuple[int] = (6,)
    _init_pose: jnp.array = jnp.array(
        [0.0, 0.0, -jnp.pi / 2.0, 0.0, 0.0, 0.0, 0.914, 0.05, 0.7, 0.0, 0.0, 0.0]
    )
    _angle_idx: int = 2
    _domain_lower = SpotSim._domain_lower
    _domain_upper = SpotSim._domain_upper

    def __init__(
        self,
        encode_angle: bool = False,
        spot_model_params: Dict = None,
        spot_obs_noise_std: jnp.array = None,
        use_obs_noise: bool = True,
        dim_goal: int = 3,
    ):
        """
        Spot robot simulator environment

        Args:
            encode_angle: Whether to encode the angle as cos(theta) and sin(theta)
            spot_model_params: Parameters for the spot dynamics model to override the default params
            spot_obs_noise_std: Observation noise standard deviation to override the default value
            use_obs_noise: Whether to use observation noise
        """
        self.dim_goal: int = dim_goal
        self.dim_state: Tuple[int] = (
            (13 + dim_goal,) if encode_angle else (12 + dim_goal,)
        )
        Dynamics.__init__(self, self.dim_state[0], 6)
        self.encode_angle: bool = encode_angle

        # initialize dynamics
        self._dynamics_model = SpotDynamicsModel(dt=self._dt, encode_angle=encode_angle)

        # set default params
        self._set_default_params()

        # check if new params are provided
        if spot_model_params is None:
            _default_params = self._default_spot_model_params
        else:
            _default_params = self._default_spot_model_params
            _default_params.update(spot_model_params)
        self._typical_params = SpotParams(**_default_params)
        self._dynamics_params = self._typical_params
        if spot_obs_noise_std is not None:
            self._obs_noise_stds = spot_obs_noise_std
        self.use_obs_noise = use_obs_noise

        # set up action buffer
        self._action_buffer = jnp.zeros((1, self.dim_action[0]))

        # initialize state
        self._init_state: jnp.array = jnp.zeros(self.dim_state)

    def init_params(self, key: chex.PRNGKey) -> SpotDynamicsParams:
        return SpotDynamicsParams(
            spot_params=self._dynamics_params,
            action_buffer=self._action_buffer,
            key=key,
        )

    def _set_default_params(self):
        from sim_transfer.sims.spot_sim_config import SPOT_DEFAULT_PARAMS
        from sim_transfer.sims.spot_sim_config import SPOT_DEFAULT_OBSERVATION_NOISE_STD

        self._default_spot_model_params = SPOT_DEFAULT_PARAMS
        self._obs_noise_stds = SPOT_DEFAULT_OBSERVATION_NOISE_STD

    def _state_to_obs(self, state: jnp.array, rng_key: chex.PRNGKey) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape == (12,), f"State shape is {state.shape} instead of 12"
        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jr.normal(rng_key, shape=state.shape)
        else:
            obs = state

        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 13 and self.encode_angle) or (
            obs.shape[-1] == 12 and not self.encode_angle
        )
        return obs

    def next_state(
        self, x: chex.Array, u: chex.Array, dynamics_params: SpotDynamicsParams
    ) -> Tuple[Distribution, SpotDynamicsParams]:
        assert u.shape == (self.u_dim,) and x.shape == (self.x_dim,)

        # split into robot state and goal
        x_state = x[: -self.dim_goal]
        x_goal = x[-self.dim_goal :]

        # get previous action and update action buffer
        prev_u = dynamics_params.action_buffer[-1]
        action_buffer = jnp.array([u[None, :]])

        # Move forward one step in the dynamics using the delayed action and the hidden state
        new_key, key_for_sampling_obs_noise = jr.split(dynamics_params.key)
        x_mean_next = self._dynamics_model.next_step(
            x_state, u, dynamics_params.spot_params
        )
        if self.encode_angle:
            x_mean_next = self._dynamics_model.reduce_x(x_mean_next)
        x_next = self._state_to_obs(x_mean_next, key_for_sampling_obs_noise)

        # add goal back to state
        x_next = jnp.concatenate([x_next, x_goal])

        new_params = dynamics_params.replace(action_buffer=action_buffer, key=new_key)
        return Normal(x_next, jnp.zeros_like(x_next)), new_params

    def reset(self, key: chex.PRNGKey) -> jnp.array:
        """Resets the environment to a random initial state close to the initial pose"""
        # sample random initial state
        reset_key, key_obs = jr.split(key, 2)

        init_state, goal = sample_pos_and_goal_spot(
            rng_key=reset_key,
            domain_lower=self._domain_lower,
            domain_upper=self._domain_upper,
        )

        return jnp.concatenate([self._state_to_obs(init_state, rng_key=key_obs), goal])


@chex.dataclass
class SpotRewardParams:
    key: chex.PRNGKey


class SpotReward(Reward[SpotRewardParams]):
    def __init__(
        self,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = False,
        bound: float = 0.1,
        margin_factor: float = 10.0,
        ctrl_diff_weight: float = 0.0,
        dim_goal: int = 3,
    ):
        Reward.__init__(
            self, x_dim=13 + dim_goal if encode_angle else 12 + dim_goal, u_dim=6
        )
        self.ctrl_diff_weight = ctrl_diff_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle: bool = encode_angle
        self.dim_goal = dim_goal
        self._reward_model = SpotEnvReward(
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            bound=bound,
            margin_factor=margin_factor,
        )

    def init_params(self, key: chex.PRNGKey) -> SpotRewardParams:
        return SpotRewardParams(key=key)

    def __call__(
        self,
        x: chex.Array,
        u: chex.Array,
        reward_params: SpotRewardParams,
        x_next: chex.Array | None = None,
    ) -> Tuple[Distribution, SpotRewardParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        assert x_next.shape == (self.x_dim,)
        reward = self._reward_model.forward(obs=None, action=u, next_obs=x_next)
        return Normal(reward, jnp.zeros_like(reward)), reward_params


class SpotSystem(System[SpotDynamicsParams, SpotRewardParams]):
    def __init__(
        self,
        encode_angle: bool = False,
        action_delay: float = 0.0,
        spot_model_params: Dict = None,
        spot_obs_noise_std: jnp.array = None,
        ctrl_cost_weight: float = 0.005,
        use_obs_noise: bool = True,
        bound: float = 0.1,
        margin_factor: float = 10.0,
        dim_goal: int = 3,
    ):
        System.__init__(
            self,
            dynamics=SpotDynamics(
                encode_angle=encode_angle,
                action_delay=action_delay,
                spot_model_params=spot_model_params,
                spot_obs_noise_std=spot_obs_noise_std,
                use_obs_noise=use_obs_noise,
                dim_goal=dim_goal,
            ),
            reward=SpotReward(
                ctrl_cost_weight=ctrl_cost_weight,
                encode_angle=encode_angle,
                bound=bound,
                margin_factor=margin_factor,
                dim_goal=dim_goal,
            ),
        )

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(
            dynamics_params=SpotDynamicsParams(
                action_buffer=axes, spot_params=None, key=axes
            ),
            reward_params=SpotRewardParams(key=axes),
            key=axes,
        )

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        x: chex.Array,
        u: chex.Array,
        system_params: SystemParams[SpotDynamicsParams, SpotRewardParams],
    ) -> SystemState:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        new_key, key_x_next, key_reward = jr.split(system_params.key, 3)
        x_next_dist, next_dynamics_params = self.dynamics.next_state(
            x, u, system_params.dynamics_params
        )
        x_next = x_next_dist.sample(seed=key_x_next)
        reward_dist, next_reward_params = self.reward(
            x, u, system_params.reward_params, x_next
        )
        return SystemState(
            x_next=x_next,
            reward=reward_dist.sample(seed=key_reward),
            system_params=SystemParams(
                dynamics_params=next_dynamics_params,
                reward_params=next_reward_params,
                key=new_key,
            ),
        )

    def reset(self, key: chex.PRNGKey) -> SystemState:
        return SystemState(
            x_next=self.dynamics.reset(key=key),
            reward=jnp.array([0.0]).squeeze(),
            system_params=self.init_params(key=key),
        )


if __name__ == "__main__":
    ENCODE_ANGLE = False
    system = SpotSystem(encode_angle=ENCODE_ANGLE, action_delay=1.0 / 10.0)
    t_start = time.time()
    system_params = system.init_params(key=jr.PRNGKey(0))
    s = system.dynamics.reset(key=jr.PRNGKey(0))

    traj = [s]
    rewards = []
    actions = []
    for i in range(120):
        t = i / 10.0
        a = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        next_sys_step = system.step(s, a, system_params)
        s = next_sys_step.x_next
        r = next_sys_step.reward
        system_params = next_sys_step.system_params
        traj.append(s)
        actions.append(a)
        rewards.append(r)

    duration = time.time() - t_start
    print(f"Duration of trajectory sim {duration} sec")
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    plot_spot_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
