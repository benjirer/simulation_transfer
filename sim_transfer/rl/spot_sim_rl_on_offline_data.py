import os
import cloudpickle as pickle
from typing import Callable, Any

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from jax import jit, vmap
from jax.lax import scan
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from sim_transfer.sims.spot_system_goal import SpotSystem
from sim_transfer.sims.envs import SpotSimEnv
from sim_transfer.sims.simulators import SpotSim
from sim_transfer.sims.util import plot_spot_trajectory
from sim_transfer.sims.util import decode_angles


class RLFromOfflineData:
    """Class to train a policy on offline data using simulator and evaluate it on the simulator."""

    def __init__(
        self,
        x_train: chex.Array,
        y_train: chex.Array,
        x_test: chex.Array,
        y_test: chex.Array,
        spot_reward_kwargs: dict = None,
        max_replay_size_true_data_buffer: int = 30**4,
        sac_kwargs: dict = None,
        key: chex.PRNGKey = jr.PRNGKey(0),
        return_best_policy: bool = True,
        test_data_ratio: float = 0.1,
        num_init_points_to_bs_for_sac_learning: int | None = 100,
        eval_sac_only_from_init_states: bool = False,
        train_sac_only_from_init_states: bool = False,
        wandb_logging: bool = True,
    ):
        # set parameters
        self.wandb_logging = wandb_logging
        self.eval_sac_only_from_init_states = eval_sac_only_from_init_states
        self.train_sac_only_from_init_states = train_sac_only_from_init_states
        self.test_data_ratio = test_data_ratio
        self.key = key
        self.return_best_policy = return_best_policy
        self.spot_reward_kwargs = spot_reward_kwargs
        self.sac_kwargs = sac_kwargs

        # split the train data into train and eval
        self.key, key_split = jr.split(self.key)
        x_train, y_train, x_eval, y_eval = self.shuffle_and_split_data(
            x_train, y_train, self.test_data_ratio, key_split
        )

        # prepare number of init points for learning
        if num_init_points_to_bs_for_sac_learning is None:
            num_init_points_to_bs_for_sac_learning = x_train.shape[0]
        self.num_init_points_to_bs_for_learning = num_init_points_to_bs_for_sac_learning

        # set dimensions
        # Note: raw x is built from [state (12/13), goal (3), action (6)] = 21/22
        state_dim = 13
        action_dim = 6
        goal_dim = 3
        state_dim_with_goal = state_dim + goal_dim
        self.state_dim = state_dim
        self.state_dim_with_goal = state_dim_with_goal
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # reshape data and prepare it for policy training
        states_obs = x_train[:, :state_dim_with_goal]
        next_state_obs = y_train
        actions = x_train[:, self.state_dim_with_goal :]

        # prepare transitions
        rewards = jnp.zeros(shape=(x_train.shape[0],))
        discounts = 0.99 * jnp.ones(shape=(x_train.shape[0],))
        transitions = Transition(
            observation=jnp.concatenate([states_obs], axis=-1),
            action=actions,
            reward=rewards,
            discount=discounts,
            next_observation=jnp.concatenate([next_state_obs], axis=-1),
        )

        # create a dummy sample to init the buffer
        dummy_obs = jnp.zeros(shape=(state_dim_with_goal,))
        self.dummy_sample = Transition(
            observation=dummy_obs,
            action=jnp.zeros(shape=(action_dim,)),
            reward=jnp.array(0.0),
            discount=jnp.array(0.99),
            next_observation=dummy_obs,
        )

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1,
        )

        # init and insert the data in the true data buffer
        self.key, key_init_buffer, key_insert_data = jr.split(self.key, 3)
        true_buffer_state = self.true_data_buffer.init(key_init_buffer)
        true_buffer_state = self.true_data_buffer.insert(true_buffer_state, transitions)
        self.true_buffer_state = true_buffer_state

    @staticmethod
    def shuffle_and_split_data(x_data, y_data, test_ratio, key: chex.PRNGKey):
        """Permute and split data into train and test sets."""
        # get the size of the data
        num_data = x_data.shape[0]

        # create a permutation of indices and permute data
        perm = jr.permutation(key, num_data)
        x_data = x_data[perm]
        y_data = y_data[perm]

        # calculate number of examples in the test set
        num_test = int(test_ratio * num_data)

        # calculate number of examples in the train set
        num_train = num_data - num_test

        # split data
        x_train = x_data[:num_train]
        x_test = x_data[num_train:]
        y_train = y_data[:num_train]
        y_test = y_data[num_train:]

        return x_train, y_train, x_test, y_test

    def prepare_init_transitions(self, key: chex.PRNGKey, number_of_samples: int):
        """Prepare initial transitions for the buffer."""
        # get simulator
        sim = SpotSimEnv(encode_angle=True)

        # prepare transitions using simulator for the initial observations
        key_init_state = jr.split(key, number_of_samples)
        state_obs = vmap(sim.reset)(rng_key=key_init_state)
        actions = jnp.zeros(shape=(number_of_samples, self.action_dim))
        rewards = jnp.zeros(shape=(number_of_samples,))
        discounts = 0.99 * jnp.ones(shape=(number_of_samples,))
        transitions = Transition(
            observation=jnp.concatenate([state_obs], axis=-1),
            action=actions,
            reward=rewards,
            discount=discounts,
            next_observation=jnp.concatenate([state_obs], axis=-1),
        )
        return transitions

    def train_policy(
        self,
        true_data_buffer_state: ReplayBufferState,
        key: chex.PRNGKey,
    ):
        """Train policy using SAC."""

        # handle keys
        key_train, key_simulate, key_init_state, *keys_sys_params = jr.split(key, 5)

        # create spot system
        system = SpotSystem(
            encode_angle=True,
            action_delay=0.00,
        )

        # add init points to the true_data_buffer
        key_init_state, key_init_buffer = jr.split(key_init_state)
        init_transitions = self.prepare_init_transitions(
            key_init_state, self.num_init_points_to_bs_for_learning
        )

        # setup training buffer
        if self.train_sac_only_from_init_states:
            train_buffer_state = self.true_data_buffer.init(key_init_buffer)
            train_buffer_state = self.true_data_buffer.insert(
                train_buffer_state, init_transitions
            )
        else:
            train_buffer_state = self.true_data_buffer.insert(
                true_data_buffer_state, init_transitions
            )

        # setup evaluation buffer
        init_states_bs = train_buffer_state
        if self.eval_sac_only_from_init_states:
            init_states_bs = self.true_data_buffer.init(key_init_buffer)
            init_states_bs = self.true_data_buffer.insert(
                init_states_bs, init_transitions
            )

        # create training and eval environments (wrap system and insert buffer and init params)
        env = BraxWrapper(
            system=system,
            sample_buffer_state=train_buffer_state,
            sample_buffer=self.true_data_buffer,
            system_params=system.init_params(keys_sys_params[0]),
        )
        eval_env = BraxWrapper(
            system=system,
            sample_buffer_state=init_states_bs,
            sample_buffer=self.true_data_buffer,
            system_params=system.init_params(keys_sys_params[0]),
        )

        # create SAC trainer
        _sac_kwargs = self.sac_kwargs
        sac_trainer = SAC(
            environment=env,
            eval_environment=eval_env,
            eval_key_fixed=True,
            return_best_model=self.return_best_policy,
            **_sac_kwargs,
        )

        # train policy
        params, metrics = sac_trainer.run_training(key=key_train)

        # get policy function
        make_inference_fn = sac_trainer.make_policy

        @jit
        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy, params, metrics

    def prepare_policy_from_offline_data(
        self,
    ):
        # train policy
        policy, params, metrics = self.train_policy(self.true_buffer_state, self.key)

        # save policy parameters
        if self.wandb_logging:
            directory = os.path.join(wandb.run.dir, "models")
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join("models", "parameters.pkl")
            with open(os.path.join(wandb.run.dir, model_path), "wb") as handle:
                pickle.dump(params, handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

        return policy, params, metrics

    @staticmethod
    def arg_mean(a: chex.Array):
        """Return index of the element that is closest to the mean."""
        return jnp.argmin(jnp.abs(a - jnp.mean(a)))

    def evaluate_policy_on_the_simulator(
        self,
        policy: Callable,
        key: chex.PRNGKey = jr.PRNGKey(0),
        num_evals: int = 1,
        save_traj_dir: str = None,
    ):
        """Evaluate policy on the simulator."""

        # get reward and trajectory on the simulator
        def reward_on_simulator(key: chex.PRNGKey):
            # init actions buffer, simulator and get initial observation
            sim = SpotSimEnv(
                encode_angle=True,
                action_delay=0.0,
                margin_factor=self.spot_reward_kwargs["margin_factor"],
                ctrl_cost_weight=self.spot_reward_kwargs["ctrl_cost_weight"],
                ctrl_diff_weight=self.spot_reward_kwargs["ctrl_diff_weight"],
                max_steps=self.sac_kwargs["episode_length"],
            )
            obs = sim.reset(key)

            # use policy on simulator
            done = False
            transitions_for_plotting = []
            while not done:
                # get action from policy
                policy_input = jnp.concatenate([obs], axis=-1)
                action = policy(policy_input)

                # step simulator
                next_obs, reward, done, info = sim.step(action)

                transitions_for_plotting.append(
                    Transition(
                        observation=obs,
                        action=action,
                        reward=jnp.array(reward),
                        discount=jnp.array(0.99),
                        next_observation=next_obs,
                    )
                )
                obs = next_obs

            concatenated_transitions_for_plotting = jtu.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *transitions_for_plotting
            )
            reward_on_simulator = jnp.sum(concatenated_transitions_for_plotting.reward)
            return reward_on_simulator, concatenated_transitions_for_plotting

        rewards, trajectories = vmap(reward_on_simulator)(jr.split(key, num_evals))

        # get mean and std of rewards
        reward_mean = jnp.mean(rewards)
        reward_std = jnp.std(rewards)

        # get trajectory with mean reward
        reward_mean_index = self.arg_mean(rewards)
        trajectory_mean = jtu.tree_map(lambda x: x[reward_mean_index], trajectories)

        # plot trajectory evaluation
        fig_eval, _ = plot_spot_trajectory(
            trajectories,
            encode_angle=True,
            plot_mode="transitions_eval_full",
        )

        # plot trajectory ee-goal distance
        fig_distance, _, mean_error_after_10_steps = plot_spot_trajectory(
            trajectories,
            encode_angle=True,
            plot_mode="transitions_distance_eval",
        )

        if self.wandb_logging:
            model_name = "simulator"
            wandb.log(
                {
                    f"Trajectory_eval_on_{model_name}": wandb.Image(fig_eval),
                    f"Distance_eval_on_{model_name}": wandb.Image(fig_distance),
                    f"mean_error_after_10_steps_on_{model_name}": float(
                        mean_error_after_10_steps
                    ),
                    f"reward_mean_on_{model_name}": float(reward_mean),
                    f"reward_std_on_{model_name}": float(reward_std),
                }
            )
            plt.close("all")

        # save trajectories
        if save_traj_dir is not None:
            model_name = "simulator"
            save_traj_dir = os.path.join(save_traj_dir, model_name)

            # save trajectories
            if not os.path.exists(save_traj_dir):
                os.makedirs(save_traj_dir)
            with open(
                os.path.join(save_traj_dir, "trajectories_all.pkl"), "wb"
            ) as handle:
                pickle.dump(trajectories, handle)
            with open(
                os.path.join(save_traj_dir, "trajectory_mean.pkl"), "wb"
            ) as handle:
                pickle.dump(trajectory_mean, handle)

            print(f"Trajectories (all and mean reward) saved in {save_traj_dir}")


if __name__ == "__main__":
    # TODO: add test
    print("Test execution not implemented.")
    pass