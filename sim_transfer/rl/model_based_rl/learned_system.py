from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution, Normal
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.dynamics.base_dynamics import Dynamics

from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.sims.car_system import CarReward, CarRewardParams, SystemState
from sim_transfer.sims.spot_system import SpotReward, SpotRewardParams


@chex.dataclass
class DynamicsParams:
    key: chex.PRNGKey


class LearnedDynamics(Dynamics[DynamicsParams]):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        model: BatchedNeuralNetworkModel,
        include_noise: bool = True,
        predict_difference: bool = True,
        num_frame_stack: int = 0,
    ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.include_noise = include_noise
        self.predict_difference = predict_difference
        self.num_frame_stack = num_frame_stack
        self._x_dim = x_dim - u_dim * num_frame_stack
        self._u_dim = u_dim

    def next_state(
        self, x: chex.Array, u: chex.Array, dynamics_params: DynamicsParams
    ) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (
            self._x_dim + self._u_dim * self.num_frame_stack,
        ) and u.shape == (self._u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        z = z.reshape((1, -1))
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        if self.predict_difference:
            delta_x_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            delta_x = delta_x_dist.sample(seed=key_sample_x_next)
            _x = x[: self._x_dim]
            _x_next = _x + delta_x.reshape((self._x_dim,))
        else:
            x_next_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            _x_next = x_next_dist.sample(seed=key_sample_x_next)
            _x_next = _x_next.reshape((self._x_dim,))

        if self.num_frame_stack > 0:
            # Update last num_frame_stack actions
            _us = x[self._x_dim :]
            new_us = jnp.concatenate([_us[self._u_dim :], u])
            x_next = jnp.concatenate([_x_next, new_us])
        else:
            x_next = _x_next

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next, scale=jnp.zeros_like(x_next)), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return DynamicsParams(key=key)


class LearnedCarSystem(System[DynamicsParams, CarRewardParams]):
    def __init__(
        self,
        model: BatchedNeuralNetworkModel,
        include_noise: bool,
        predict_difference: bool,
        num_frame_stack: int = 0,
        **car_reward_kwargs: dict
    ):
        self.num_frame_stack = num_frame_stack
        reward = CarReward(**car_reward_kwargs, num_frame_stack=num_frame_stack)
        dynamics = LearnedDynamics(
            x_dim=reward.x_dim + self.num_frame_stack * reward.u_dim,
            u_dim=reward.u_dim,
            model=model,
            include_noise=include_noise,
            predict_difference=predict_difference,
            num_frame_stack=num_frame_stack,
        )
        System.__init__(self, dynamics=dynamics, reward=reward)
        self._x_dim = reward.x_dim
        self._u_dim = reward.u_dim

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(
            dynamics_params=DynamicsParams(key=axes),
            reward_params=CarRewardParams(_goal=None, key=axes),
            key=axes,
        )

    def step(
        self,
        x: chex.Array,
        u: chex.Array,
        system_params: SystemParams[DynamicsParams, CarRewardParams],
    ) -> SystemState:
        assert x.shape == (
            self._x_dim + self._u_dim * self.num_frame_stack,
        ) and u.shape == (self._u_dim,)
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


class LearnedSpotDynamics(Dynamics[DynamicsParams]):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        model: BatchedNeuralNetworkModel,
        include_noise: bool = True,
        predict_difference: bool = True,
        num_frame_stack: int = 0,
        dim_goal: int = 3,
    ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.include_noise = include_noise
        self.predict_difference = predict_difference
        self.num_frame_stack = num_frame_stack
        self._x_dim = x_dim - u_dim * num_frame_stack
        self._u_dim = u_dim
        self._dim_goal = dim_goal
        self._x_dim_no_goal = self._x_dim - self._dim_goal

    def next_state(
        self, x_raw: chex.Array, u: chex.Array, dynamics_params: DynamicsParams
    ) -> Tuple[Distribution, DynamicsParams]:
        assert x_raw.shape == (
            self._x_dim + self._u_dim * self.num_frame_stack,
        ) and u.shape == (self._u_dim,)

        # # print dims
        # print("self.x_dim", self.x_dim)
        # print("self._x_dim", self._x_dim)
        # print("self._u_dim", self._u_dim)
        # print("self.num_frame_stack", self.num_frame_stack)
        # print("self._dim_goal", self._dim_goal)
        # print("self._x_dim_no_goal", self._x_dim_no_goal)

        # remove goal from state
        print("x_raw shape", x_raw.shape)
        x_raw_unaugmented = x_raw[: self._x_dim] # dim = 16
        frame_stack = x_raw[self._x_dim : self._x_dim + self._u_dim * self.num_frame_stack] # dim = 12
        goal = x_raw_unaugmented[self._x_dim_no_goal : self._x_dim] # dim = 3
        x_unaugmented_no_goal = x_raw_unaugmented[: self._x_dim_no_goal] # dim = 13
        x = jnp.concatenate([x_unaugmented_no_goal, frame_stack]) # dim = 13 + 12 = 25
        print("x shape", x.shape)
        assert x.shape == (
            self._x_dim_no_goal + self._u_dim * self.num_frame_stack,
        ), "x shape is wrong, expected {}, got {}".format(
            (self._x_dim_no_goal + self._u_dim * self.num_frame_stack,), x.shape
        )
        assert goal.shape == (self._dim_goal,)

        # create state-action pair
        z = jnp.concatenate([x, u]) # dim = 25 + 3 = 28
        z = z.reshape((1, -1))

        # predict next state
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        if self.predict_difference:
            delta_x_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            delta_x = delta_x_dist.sample(seed=key_sample_x_next)
            _x = x[: self._x_dim_no_goal]
            _x_next = _x + delta_x.reshape((self._x_dim_no_goal,))
        else:
            x_next_dist = self.model.predict_dist(z, include_noise=self.include_noise)
            _x_next = x_next_dist.sample(seed=key_sample_x_next)
            _x_next = _x_next.reshape((self._x_dim_no_goal,))

        # add goal back to state
        _x_next = jnp.concatenate([_x_next, goal]) # dim = 16

        if self.num_frame_stack > 0:
            # update last num_frame_stack actions
            _us = x_raw[self._x_dim :] # dim = 12
            new_us = jnp.concatenate([_us[self._u_dim :], u]) # dim = 12
            x_next = jnp.concatenate([_x_next, new_us]) # dim = 16 + 12 = 28
        else:
            x_next = _x_next

        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next, scale=jnp.zeros_like(x_next)), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return DynamicsParams(key=key)


class LearnedSpotSystem(System[DynamicsParams, SpotRewardParams]):
    def __init__(
        self,
        model: BatchedNeuralNetworkModel,
        include_noise: bool,
        predict_difference: bool,
        num_frame_stack: int = 0,
        dim_goal: int = 3,
        **spot_reward_kwargs: dict
    ):
        reward = SpotReward(
            **spot_reward_kwargs, num_frame_stack=num_frame_stack, dim_goal=dim_goal
        )
        dynamics = LearnedSpotDynamics(
            x_dim=reward.x_dim + num_frame_stack * reward.u_dim,
            u_dim=reward.u_dim,
            dim_goal=dim_goal,
            model=model,
            include_noise=include_noise,
            predict_difference=predict_difference,
            num_frame_stack=num_frame_stack,
        )
        System.__init__(self, dynamics=dynamics, reward=reward)
        self.num_frame_stack = num_frame_stack
        self._x_dim = reward.x_dim
        self._u_dim = reward.u_dim
        self._dim_goal = dim_goal

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(
            dynamics_params=DynamicsParams(key=axes),
            reward_params=SpotRewardParams(key=axes),
            key=axes,
        )

    def step(
        self,
        x: chex.Array,
        u: chex.Array,
        system_params: SystemParams[DynamicsParams, SpotRewardParams],
    ) -> SystemState:
        assert x.shape == (
            self._x_dim + self.num_frame_stack * self._u_dim,
        ) and u.shape == (self._u_dim,)

        # handle keys
        new_key, key_x_next, key_reward = jr.split(system_params.key, 3)

        # get next state
        x_next_dist, next_dynamics_params = self.dynamics.next_state(
            x, u, system_params.dynamics_params
        )
        x_next = x_next_dist.sample(seed=key_x_next)

        # get reward
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
