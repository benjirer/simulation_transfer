import os
import cloudpickle as pickle
from typing import Callable, Any
import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import jit, vmap
from jax.lax import scan
import matplotlib.pyplot as plt
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition

from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.brax_wrapper import BraxWrapper

from experiments.data_provider import provide_data_and_sim, _SPOT_NOISE_STD_ENCODED
from experiments.data_provider import _load_spot_datasets
from experiments.data_provider import stack_spot_actions
from sim_transfer.sims.spot_system import SpotSystem
from sim_transfer.sims.envs import SpotSimEnv
from sim_transfer.sims.simulators import SpotSim
from sim_transfer.sims.util import plot_spot_trajectory
from sim_transfer.models.abstract_model import BatchedNeuralNetworkModel
from sim_transfer.models.bnn_svgd import BNN_SVGD
from sim_transfer.rl.model_based_rl.learned_system import LearnedSpotSystem
from sim_transfer.sims.dynamics_models import SpotParams
from sim_transfer.sims.util import encode_angles_spot as encode_angles_spot_fn
from sim_transfer.sims.util import decode_angles_spot as decode_angles_spot_fn
from sim_transfer.sims.spot_sim_config import (
    SPOT_STATE_LENGTH,
    SPOT_STATE_LENGTH_ENCODED,
    SPOT_ACTION_LENGTH,
    SPOT_GOAL_LENGTH,
    SPOT_ANGLE_IDX,
    SPOT_STATE_LABELS,
    SPOT_STATE_LABELS_ENCODED,
    SPOT_ACTION_LABELS,
    SPOT_GOAL_LABELS,
)


class RLFromOfflineData:
    """Class to train a policy for Spot on offline data."""

    def __init__(
        # parameters general
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
        num_offline_collected_transitions: int = 1000,
        test_data_ratio: float = 0.1,
        num_frame_stack: int = 0,
        num_init_points_to_bs_for_sac_learning: int | None = 100,
        eval_sac_only_from_init_states: bool = False,
        train_sac_only_from_init_states: bool = False,
        predict_difference: bool = True,
        wandb_logging: bool = True,
        # parameters model
        bnn_model: BatchedNeuralNetworkModel = None,
        include_aleatoric_noise: bool = True,
        eval_bnn_model_on_all_offline_data: bool = True,
        load_pretrained_bnn_model: bool = False,
        num_sim_fitting_steps: int = 40_000,
    ):
        # set parameters general
        self.wandb_logging = wandb_logging
        self.eval_sac_only_from_init_states = eval_sac_only_from_init_states
        self.train_sac_only_from_init_states = train_sac_only_from_init_states
        self.num_offline_collected_transitions = num_offline_collected_transitions
        self.test_data_ratio = test_data_ratio
        self.key = key
        self.return_best_policy = return_best_policy
        self.spot_reward_kwargs = spot_reward_kwargs
        self.sac_kwargs = sac_kwargs
        self.spot_learned_params = None  # setting none forces use of default params
        self.spot_learned_observation_noise_std = (
            None  # setting none forces use of default params
        )
        self.predict_difference = predict_difference

        # set parameters model
        self.load_pretrained_bnn_model = load_pretrained_bnn_model
        self.include_aleatoric_noise = include_aleatoric_noise
        self.bnn_model = bnn_model
        self.evaluate_bnn_model_on_all_collected_data = (
            eval_bnn_model_on_all_offline_data
        )
        self.num_sim_fitting_steps = num_sim_fitting_steps

        # load pretrained model for evaluation
        if self.load_pretrained_bnn_model:
            simulation_transfer_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            bnn_dir = os.path.join(simulation_transfer_dir, "bnn_models_pretrained")
            bnn_model_path = os.path.join(bnn_dir, "bnn_svgd_model_on_5_000_points.pkl")

            with open(bnn_model_path, "rb") as handle:
                bnn_model_pretrained = pickle.load(handle)

            self.bnn_model_pretrained = bnn_model_pretrained
        else:
            self.bnn_model_pretrained = None

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
        self.state_dim = SPOT_STATE_LENGTH_ENCODED
        self.action_dim = SPOT_ACTION_LENGTH
        self.goal_dim = SPOT_GOAL_LENGTH
        self.state_dim_with_goal = self.state_dim + self.goal_dim

        # account for frame stacking (augmenting state with actions)
        self.num_frame_stack = num_frame_stack
        self.state_dim_frame_stacked = (
            self.state_dim + num_frame_stack * self.action_dim
        )
        self.state_dim_with_goal_frame_stacked = (
            self.state_dim_with_goal + self.num_frame_stack * self.action_dim
        )

        # reshape data and prepare it for policy training
        states_obs = x_train[:, : self.state_dim_with_goal]
        next_state_obs = y_train[:, : self.state_dim_with_goal]
        last_actions = x_train[:, self.state_dim_with_goal_frame_stacked :]
        framestacked_actions = x_train[
            :, self.state_dim_with_goal : self.state_dim_with_goal_frame_stacked
        ]

        if self.num_frame_stack > 0:
            next_framestacked_actions = x_train[
                :, self.state_dim_with_goal + self.action_dim :
            ]
        else:
            next_framestacked_actions = framestacked_actions

        # prepare transitions
        rewards = jnp.zeros(shape=(x_train.shape[0],))
        discounts = 0.99 * jnp.ones(shape=(x_train.shape[0],))
        transitions = Transition(
            observation=jnp.concatenate([states_obs, framestacked_actions], axis=-1),
            action=last_actions,
            reward=rewards,
            discount=discounts,
            next_observation=jnp.concatenate(
                [next_state_obs, next_framestacked_actions], axis=-1
            ),
        )

        # create a dummy sample to init the buffer
        dummy_obs = jnp.zeros(shape=(self.state_dim_with_goal_frame_stacked,))
        self.dummy_sample = Transition(
            observation=dummy_obs,
            action=jnp.zeros(shape=(self.action_dim,)),
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

        # prepare data to train the model
        # note: we have to remove the goal from the data but keep the augmented state
        # x data
        x_train, u_train = (
            jnp.concatenate(
                [
                    x_train[:, : self.state_dim],
                    x_train[
                        :,
                        self.state_dim_with_goal : self.state_dim_with_goal_frame_stacked,
                    ],
                ],
                axis=-1,
            ),
            x_train[:, self.state_dim_with_goal_frame_stacked :],
        )
        x_eval, u_eval = (
            jnp.concatenate(
                [
                    x_eval[:, : self.state_dim],
                    x_eval[
                        :,
                        self.state_dim_with_goal : self.state_dim_with_goal_frame_stacked,
                    ],
                ],
                axis=-1,
            ),
            x_eval[:, self.state_dim_with_goal_frame_stacked :],
        )
        x_test, u_test = (
            jnp.concatenate(
                [
                    x_test[:, : self.state_dim],
                    x_test[
                        :,
                        self.state_dim_with_goal : self.state_dim_with_goal_frame_stacked,
                    ],
                ],
                axis=-1,
            ),
            x_test[:, self.state_dim_with_goal_frame_stacked :],
        )
        x_train = jnp.concatenate([x_train, u_train], axis=-1)
        x_eval = jnp.concatenate([x_eval, u_eval], axis=-1)
        x_test = jnp.concatenate([x_test, u_test], axis=-1)

        # y data
        y_train = y_train[:, : self.state_dim]
        y_eval = y_eval[:, : self.state_dim]
        y_test = y_test[:, : self.state_dim]
        print(
            "Data adapted for model training: x_train.shape",
            x_train.shape,
            "x_eval.shape",
            x_eval.shape,
            "x_test.shape",
            x_test.shape,
            "y_train.shape",
            y_train.shape,
            "y_eval.shape",
            y_eval.shape,
            "y_test.shape",
            y_test.shape,
        )

        assert (
            x_train.shape[-1]
            == self.state_dim + (self.num_frame_stack + 1) * self.action_dim
            and x_eval.shape[-1]
            == self.state_dim + (self.num_frame_stack + 1) * self.action_dim
            and x_test.shape[-1]
            == self.state_dim + (self.num_frame_stack + 1) * self.action_dim
            and y_train.shape[-1] == self.state_dim
            and y_eval.shape[-1] == self.state_dim
            and y_test.shape[-1] == self.state_dim
        ), "Model training data has wrong shape."

        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.x_test = x_test
        self.y_test = y_test

        # prepare data for predict_difference mode
        if self.predict_difference:
            self.y_train = self.y_train - self.x_train[..., : self.state_dim]
            self.y_eval = self.y_eval - self.x_eval[..., : self.state_dim]
            self.y_test = self.y_test - self.x_test[..., : self.state_dim]

    """============== Model training function =============="""

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

    def visualize_data(self, output_dir="plots", max_timesteps=None):
        """
        Visualizes the time series data by plotting all state and action variables
        over time in one figure using subplots arranged in two columns,
        matching action variables with corresponding state variables.

        Args:
            output_dir (str): Directory where the plot will be saved.
            max_timesteps (int or None): Maximum number of timesteps to plot. If None, plots all timesteps.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_steps_train = self.x_train.shape[0]

        # If max_timesteps is specified, limit the number of timesteps
        if max_timesteps is not None:
            time_steps = min(time_steps_train, max_timesteps)
            x_train = self.x_train[:time_steps]
            y_train = self.y_train[:time_steps]
        else:
            time_steps = time_steps_train
            x_train = self.x_train
            y_train = self.y_train

        # Define state and action labels
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

        num_state_vars = len(state_labels)
        num_action_vars = len(action_labels)
        num_frame_stack = self.num_frame_stack  # Assuming this attribute exists

        # Total number of action variables in input: (1 + num_frame_stack) * num_action_vars
        total_action_vars = (1 + num_frame_stack) * num_action_vars

        # x_train is of shape [time_steps, num_state_vars + total_action_vars]
        # Extract state variables
        state_data = x_train[:, :num_state_vars]

        # Extract action variables
        action_data = x_train[:, num_state_vars : num_state_vars + total_action_vars]

        # Create a mapping from action labels to indices
        action_label_to_index = {label: idx for idx, label in enumerate(action_labels)}

        # Prepare the figure
        num_rows = num_state_vars  # One row per state variable
        num_cols = 2  # One column for states, one for actions

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(12, num_rows * 2), sharex=True
        )
        time = range(time_steps)

        # Plot state variables in the first column
        for idx in range(num_state_vars):
            axes[idx, 0].plot(time, state_data[:, idx])
            axes[idx, 0].set_ylabel(state_labels[idx])
            if idx == 0:
                axes[idx, 0].set_title("State Variables")
            if idx == num_state_vars - 1:
                axes[idx, 0].set_xlabel("Time Step")

        # split stacked actions into individual actions, for now lets assume num_frame_stack = 2
        first_action_data = action_data[:, :num_action_vars]
        second_action_data = action_data[:, num_action_vars : 2 * num_action_vars]
        # third_action_data = action_data[:, 2*num_action_vars:3*num_action_vars]

        # Plot corresponding action variables in the second column
        for idx in range(num_state_vars):
            state_label = state_labels[idx]
            if state_label in action_labels:
                action_idx = action_label_to_index[state_label]
                axes[idx, 1].plot(time, first_action_data[:, action_idx], label="t")
                # axes[idx, 1].plot(time, second_action_data[:, action_idx], label='t+1')
                # axes[idx, 1].plot(time, third_action_data[:, action_idx], label='t+2')
                axes[idx, 1].set_ylabel(state_label)
                axes[idx, 1].legend()
            else:
                # If no corresponding action, turn off the subplot
                axes[idx, 1].axis("off")

            if idx == 0:
                axes[idx, 1].set_title("Action Variables")
            if idx == num_state_vars - 1:
                axes[idx, 1].set_xlabel("Time Step")

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_series_plot.png"))
        plt.close()

        print(
            f'Time series plot saved as "time_series_plot.png" in the "{output_dir}" directory.'
        )

    def train_model(
        self, bnn_train_steps: int, return_best_bnn: bool = True
    ) -> BNN_SVGD:
        """Train selected model on the collected data."""

        # get data
        x_train, y_train, x_eval, y_eval = (
            self.x_train,
            self.y_train,
            self.x_eval,
            self.y_eval,
        )

        x_test, y_test = self.x_test, self.y_test

        # create bnn model
        bnn = self.bnn_model

        # set evaluation mode
        if self.test_data_ratio == 0.0:
            metrics_objective = "train_nll_loss"
            x_eval, y_eval = x_test, y_test
        else:
            metrics_objective = "eval_nll"

        # confirm shape
        print(
            "Training bnn model with:",
            "x_train.shape",
            x_train.shape,
            "y_train.shape",
            y_train.shape,
            "x_eval.shape",
            x_eval.shape,
            "y_eval.shape",
            y_eval.shape,
        )

        # train bnn model
        bnn.fit(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            log_to_wandb=True,
            keep_the_best=return_best_bnn,
            metrics_objective=metrics_objective,
            num_steps=bnn_train_steps,
        )
        return bnn

    """============== Policy training functions =============="""

    def prepare_init_transitions(self, key: chex.PRNGKey, number_of_samples: int):
        """Prepare initial transitions for the buffer."""
        # get simulator
        sim = SpotSimEnv(encode_angle=True)

        # prepare transitions using simulator for the initial observations
        key_init_state = jr.split(key, number_of_samples)
        state_obs = vmap(sim.reset)(rng_key=key_init_state)
        framestacked_actions = jnp.zeros(
            shape=(number_of_samples, self.num_frame_stack * self.action_dim)
        )
        actions = jnp.zeros(shape=(number_of_samples, self.action_dim))
        rewards = jnp.zeros(shape=(number_of_samples,))
        discounts = 0.99 * jnp.ones(shape=(number_of_samples,))
        transitions = Transition(
            observation=jnp.concatenate([state_obs, framestacked_actions], axis=-1),
            action=actions,
            reward=rewards,
            discount=discounts,
            next_observation=jnp.concatenate(
                [state_obs, framestacked_actions], axis=-1
            ),
        )
        return transitions

    def train_policy(
        self,
        bnn_model: BatchedNeuralNetworkModel,
        true_data_buffer_state: ReplayBufferState,
        key: chex.PRNGKey,
    ):
        """Train policy using SAC and BNN-MODEL."""

        # handle keys
        key_train, key_simulate, key_init_state, *keys_sys_params = jr.split(key, 5)

        # create a learned spot system
        system = LearnedSpotSystem(
            model=bnn_model,
            include_noise=self.include_aleatoric_noise,
            predict_difference=self.predict_difference,
            num_frame_stack=self.num_frame_stack,
            **self.spot_reward_kwargs,
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

    def prepare_policy(self, params: Any | None = None, filename: str = None):
        """Prepare policy function for inference from parameters or file using BNN-MODEL."""

        # load params from file if not provided directly
        if params is None:
            with open(filename, "rb") as handle:
                params = pickle.load(handle)

        # get data (only used for shape)
        x_train, y_train, x_test, y_test = (
            self.x_train,
            self.y_train,
            self.x_eval,
            self.y_eval,
        )

        # create a bnn model
        standard_model_params = {
            "input_size": x_train.shape[-1],
            "output_size": y_train.shape[-1],
            "rng_key": jr.PRNGKey(234234345),
            # 'normalization_stats': sim.normalization_stats, TODO: Jonas: adjust sim for normalization stats
            "likelihood_std": _SPOT_NOISE_STD_ENCODED,
            "normalize_likelihood_std": True,
            "learn_likelihood_std": True,
            "likelihood_exponent": 0.5,
            "hidden_layer_sizes": [64, 64, 64],
            "data_batch_size": 32,
        }
        bnn = BNN_SVGD(**standard_model_params, bandwidth_svgd=1.0)

        # create learned spot system
        system = LearnedSpotSystem(
            model=bnn,
            include_noise=self.include_aleatoric_noise,
            predict_difference=self.predict_difference,
            num_frame_stack=self.num_frame_stack,
            **self.spot_reward_kwargs,
        )

        # handle keys
        key_train, key_simulate, *keys_sys_params = jr.split(self.key, 4)

        # create env
        env = BraxWrapper(
            system=system,
            sample_buffer_state=self.true_buffer_state,
            sample_buffer=self.true_data_buffer,
            system_params=system.init_params(keys_sys_params[0]),
        )

        # create SAC trainer
        _sac_kwargs = self.sac_kwargs
        sac_trainer = SAC(
            environment=env,
            eval_environment=env,
            eval_key_fixed=True,
            return_best_model=self.return_best_policy,
            **_sac_kwargs,
        )

        # get policy function
        make_inference_fn = sac_trainer.make_policy

        @jit
        def policy(x):
            return make_inference_fn(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        return policy

    def eval_bnn_model_on_all_collected_data(
        self, bnn_model: BatchedNeuralNetworkModel
    ):
        """Evaluate learned model on all collected data."""

        # get data
        data_source: str = "spot_real"
        data_spec: dict = {
            "num_samples_train": 4_000,
            "num_stacked_actions": self.num_frame_stack,
        }
        x_data, y_data, _, _, sim = provide_data_and_sim(
            data_source=data_source, data_spec=data_spec
        )

        # prepare data for predict_difference mode
        if self.predict_difference:
            y_data = y_data - x_data[..., : self.state_dim]

        # evaluate bnn model
        eval_stats = bnn_model.eval(
            x_data, y_data, per_dim_metrics=True, prefix="eval_on_all_offline_data/"
        )
        if self.wandb_logging:
            wandb.log(eval_stats)

    def eval_bnn_model_on_test_data(self, bnn_model: BatchedNeuralNetworkModel):
        """Evaluate learned model on test data."""

        # get data
        x_test, y_test = self.x_test, self.y_test

        # evaluate bnn model
        test_stats = bnn_model.eval(
            x_test, y_test, per_dim_metrics=True, prefix="test_data/"
        )
        if self.wandb_logging:
            wandb.log(test_stats)

    def prepare_policy_from_offline_data(
        self, bnn_train_steps: int = 10_000, return_best_bnn: bool = True
    ):
        """Prepare policy from offline data using BNN-MODEL."""

        # train bnn model
        bnn_model = self.train_model(
            bnn_train_steps=bnn_train_steps, return_best_bnn=return_best_bnn
        )

        # save bnn model
        if self.wandb_logging:
            directory = os.path.join(wandb.run.dir, "models")
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join("models", "bnn_model.pkl")
            with open(os.path.join(wandb.run.dir, model_path), "wb") as handle:
                pickle.dump(bnn_model, handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

        # evaluate bnn model on all collected data
        if self.evaluate_bnn_model_on_all_collected_data:
            self.eval_bnn_model_on_all_collected_data(bnn_model)

        # train policy
        policy, params, metrics = self.train_policy(
            bnn_model, self.true_buffer_state, self.key
        )

        # save policy parameters
        if self.wandb_logging:
            directory = os.path.join(wandb.run.dir, "models")
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join("models", "parameters.pkl")
            with open(os.path.join(wandb.run.dir, model_path), "wb") as handle:
                pickle.dump(params, handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

        return policy, params, metrics, bnn_model

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
        """Evaluate policy on the default SIM-MODEL."""

        # get reward and trajectory on the simulator
        def reward_on_simulator(key: chex.PRNGKey):
            # init actions buffer, simulator and get initial observation
            actions_buffer = jnp.zeros(shape=(self.action_dim * self.num_frame_stack))
            sim = SpotSimEnv(
                encode_angle=True,
                action_delay=1 / 15.0 * self.num_frame_stack,
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
                policy_input = jnp.concatenate([obs, actions_buffer], axis=-1)
                action = policy(policy_input)

                # step simulator
                next_obs, reward, done, info = sim.step(action)

                # handle action buffer
                if self.num_frame_stack > 0:
                    next_actions_buffer = jnp.concatenate(
                        [actions_buffer[self.action_dim :], action]
                    )
                else:
                    next_actions_buffer = jnp.zeros(shape=(0,))

                transitions_for_plotting.append(
                    Transition(
                        observation=obs,
                        action=action,
                        reward=jnp.array(reward),
                        discount=jnp.array(0.99),
                        next_observation=next_obs,
                    )
                )
                actions_buffer = next_actions_buffer
                obs = next_obs

            concatenated_transitions_for_plotting = jtu.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *transitions_for_plotting
            )
            reward_on_simulator = jnp.sum(concatenated_transitions_for_plotting.reward)
            return reward_on_simulator, concatenated_transitions_for_plotting

        # get rewards and trajectories
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
            plot_mode="transitions_eval_full",
        )

        # plot trajectory ee-goal distance
        fig_distance, _, mean_error_after_10_steps = plot_spot_trajectory(
            trajectories,
            plot_mode="transitions_distance_eval",
        )

        if self.wandb_logging:
            model_name = "default_sim_model"
            wandb.log(
                {
                    f"Trajectory_eval_on_{model_name}": wandb.Image(fig_eval),
                    f"Distance_eval_on_{model_name}": wandb.Image(fig_distance),
                    f"mean_pos_error_after_10_steps_on_{model_name}": float(
                        mean_error_after_10_steps[0]
                    ),
                    f"mean_orient_error_after_10_steps_on_{model_name}": float(
                        mean_error_after_10_steps[1]
                    ),
                    f"reward_mean_on_{model_name}": float(reward_mean),
                    f"reward_std_on_{model_name}": float(reward_std),
                }
            )
            plt.close("all")

        # save trajectories
        if save_traj_dir is not None:
            model_name = "default_sim_model"
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

    def evaluate_policy(
        self,
        policy: Callable,
        bnn_model: BatchedNeuralNetworkModel | None = None,
        key: chex.PRNGKey = jr.PRNGKey(0),
        num_evals: int = 1,
        save_traj_dir: str = None,
    ):
        """Evaluate policy on the learned model"""

        # set parameters
        eval_horizon = self.sac_kwargs["episode_length"]
        model_name = "pretrained_model" if bnn_model is None else "learned_model"
        init_stacked_actions = jnp.zeros(
            shape=(self.num_frame_stack * self.action_dim,)
        )

        # create simulator
        sim = SpotSimEnv(encode_angle=True)

        # handle keys
        key_init_obs, key_generate_trajectories = jr.split(key)
        key_generate_trajectories = jr.split(key_generate_trajectories, num_evals)
        key_init_obs = jr.split(key_init_obs, num_evals)

        # get initial observations from simulator
        obs = vmap(sim.reset)(rng_key=key_init_obs)

        # get bnn model
        if bnn_model is None:
            bnn_model = self.bnn_model_pretrained
            if bnn_model is None:
                raise ValueError("You have not loaded the pretrained model.")

        # set up learned spot system
        learned_spot_system = LearnedSpotSystem(
            model=bnn_model,
            include_noise=self.include_aleatoric_noise,
            predict_difference=self.predict_difference,
            num_frame_stack=self.num_frame_stack,
            **self.spot_reward_kwargs,
        )

        # simulation step
        def f_step(carry, _):
            state, sys_params = carry
            action = policy(state)
            sys_state = learned_spot_system.step(
                x=state, u=action, system_params=sys_params
            )
            new_state = sys_state.x_next
            transition = Transition(
                observation=state[: self.state_dim_with_goal],
                action=action,
                reward=sys_state.reward,
                discount=jnp.array(0.99),
                next_observation=new_state[: self.state_dim_with_goal],
            )
            new_carry = (new_state, sys_state.system_params)
            return new_carry, transition

        # simulation loop
        def get_trajectory_transitions(init_obs, key):
            sys_params = learned_spot_system.init_params(key)
            state = jnp.concatenate([init_obs, init_stacked_actions], axis=-1)
            last_carry, transitions = scan(
                f_step, (state, sys_params), None, length=eval_horizon
            )
            return transitions

        # get trajectories
        trajectories = vmap(get_trajectory_transitions)(obs, key_generate_trajectories)

        # get rewards
        rewards = jnp.sum(trajectories.reward, axis=-1)

        # get mean and std of rewards
        reward_mean = jnp.mean(rewards)
        reward_std = jnp.std(rewards)

        # get trajectory with mean reward
        reward_mean_index = self.arg_mean(rewards)
        trajectory_mean = jtu.tree_map(lambda x: x[reward_mean_index], trajectories)

        # plot trajectory evaluation
        fig, axes = plot_spot_trajectory(
            trajectories,
            plot_mode="transitions_eval_full",
        )

        # plot trajectory ee-goal distance
        fig_distance, _, mean_error_after_10_steps = plot_spot_trajectory(
            trajectories,
            plot_mode="transitions_distance_eval",
        )

        if self.wandb_logging:
            wandb.log(
                {
                    f"Trajectory_eval_on_{model_name}": wandb.Image(fig),
                    f"Distance_eval_on_{model_name}": wandb.Image(fig_distance),
                    f"mean_pos_error_after_10_steps_on_{model_name}": float(
                        mean_error_after_10_steps[0]
                    ),
                    f"mean_orient_error_after_10_steps_on_{model_name}": float(
                        mean_error_after_10_steps[1]
                    ),
                    f"reward_mean_on_{model_name}": float(reward_mean),
                    f"reward_std_on_{model_name}": float(reward_std),
                }
            )
        plt.close("all")

        # save trajectories
        if save_traj_dir is not None:
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

    def eval_model_on_dedicated_data(
        self,
        bnn_model: BatchedNeuralNetworkModel = None,
        use_all_data: bool = False,
        step_range: int = 150,
        use_sim_data: bool = False,
    ):
        """Evaluate model on the dedicated set of data."""

        # load measured data for testing and eval
        DATA_DIR = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
        )
        if use_all_data:
            dir_path = (
                os.path.join(DATA_DIR, "recordings_spot_ee_v0"),
                os.path.join(DATA_DIR, "recordings_spot_ee_v1"),
                os.path.join(DATA_DIR, "recordings_spot_ee_v2"),
                os.path.join(DATA_DIR, "test_data_spot_ee"),
            )
            eval_trajectories_paths = sorted(
                [
                    os.path.join(dir_path, f)
                    for d in dir_path
                    for f in os.listdir(d)
                    if f.endswith(".pickle")
                ]
            )
        else:
            dir_path = os.path.join(DATA_DIR, "test_data_spot_ee")
            eval_trajectories_paths = sorted(
                [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.endswith(".pickle")
                ]
            )
        eval_trajectories, eval_trajectories_id = _load_spot_datasets(
            eval_trajectories_paths
        ), [os.path.basename(f).split(".")[0] for f in eval_trajectories_paths]

        # setup extra eval metrics (rmse over all trajectories)
        extra_eval_metrics = {}

        # iterate over eval trajectories
        for traj, traj_id in zip(eval_trajectories, eval_trajectories_id):
            # unpack data
            testing_x_pre_org = jnp.array([t.observation for t in traj])
            testing_u_pre_org = jnp.array([t.action for t in traj])
            testing_y_org = jnp.array([t.next_observation for t in traj])
            testing_y_org_raw = testing_y_org

            # decode angles as they are encoded in raw data
            testing_x_pre_org = decode_angles_spot_fn(testing_x_pre_org, SPOT_ANGLE_IDX)
            testing_y_org = decode_angles_spot_fn(testing_y_org, SPOT_ANGLE_IDX)

            # cut data
            testing_x_pre = testing_x_pre_org[:step_range]
            testing_u_pre = testing_u_pre_org[:step_range]
            testing_y = testing_y_org[:step_range]
            testing_y_org_raw = testing_y_org_raw[:step_range]

            # apply action delay
            testing_u_pre = stack_spot_actions(
                u=testing_u_pre,
                num_stacked_actions=self.num_frame_stack,
                action_dim=self.action_dim,
            )

            # prepare data
            testing_x_pre = encode_angles_spot_fn(testing_x_pre, SPOT_ANGLE_IDX)
            testing_x = jnp.concatenate([testing_x_pre, testing_u_pre], axis=1)

            from experiments.data_provider import sample_from_spot_sim

            # get data from sim
            if use_sim_data:
                testing_x, testing_y, _, _, sim = sample_from_spot_sim(
                    num_samples=120,
                    num_stacked_actions=self.num_frame_stack,
                    eval_directly=False,
                )

                testing_y = decode_angles_spot_fn(testing_y, SPOT_ANGLE_IDX)
                testing_u_pre = testing_x[:, 19:]

            # prepare error metrics
            model_errors_max_ee = {}
            model_errors_total_ee = {}
            model_errors_max_base = {}
            model_errors_total_base = {}
            model_errors_max_base_theta = {}
            model_errors_total_base_theta = {}
            model_errors_max_ee_orient = {}
            model_errors_total_ee_orient = {}

            # simulate trajectory with learned model
            model = bnn_model
            model_name = "bnn_model"
            y_pred_testing = []
            x_state = testing_x[0:1]
            for i in range(testing_x.shape[0]):
                if self.predict_difference:
                    delta_x, _ = model.predict(x_state)
                    # reshape delta_x (a tuple) to match the shape of x_state
                    print(delta_x.shape)
                    y_pred = x_state[..., : self.state_dim] + delta_x
                else:
                    y_pred, _ = model.predict(x_state)
                y_pred_testing.append(y_pred[0])
                if i < testing_x.shape[0] - 1:
                    u_next = testing_u_pre[i + 1 : i + 2]
                    x_state = jnp.concatenate([y_pred, u_next], axis=1)
            y_pred_testing = jnp.array(y_pred_testing)
            # else:
            #     raise ValueError(
            #         "Either spot_learned_params or bnn_model has to be provided."
            #     )

            y_pred_testing_raw = y_pred_testing
            y_pred_testing = decode_angles_spot_fn(y_pred_testing, SPOT_ANGLE_IDX)

            # # calculate errors for plotting
            # ee_pos_error_running = jnp.linalg.norm(
            #     y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1
            # )
            # ee_pos_error_cumulative = jnp.cumsum(ee_pos_error_running)
            # ee_pos_error_max = jnp.max(ee_pos_error_running)
            # ee_pos_error_total = jnp.sum(ee_pos_error_running)
            # model_errors_max_ee[model_name] = ee_pos_error_max
            # model_errors_total_ee[model_name] = ee_pos_error_total

            # base_pos_error_running = jnp.linalg.norm(
            #     y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1
            # )
            # base_pos_error_cumulative = jnp.cumsum(base_pos_error_running)
            # base_pos_error_max = jnp.max(base_pos_error_running)
            # base_pos_error_total = jnp.sum(base_pos_error_running)
            # model_errors_max_base[model_name] = base_pos_error_max
            # model_errors_total_base[model_name] = base_pos_error_total

            # base_theta_error_running = jnp.linalg.norm(
            #     y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1
            # )
            # base_theta_error_cumulative = jnp.cumsum(base_theta_error_running)
            # base_theta_error_max = jnp.max(base_theta_error_running)
            # base_theta_error_total = jnp.sum(base_theta_error_running)
            # model_errors_max_base_theta[model_name] = base_theta_error_max
            # model_errors_total_base_theta[model_name] = base_theta_error_total

            # ee_orient_error_running = jnp.linalg.norm(
            #     y_pred_testing[:, 12:15] - testing_y[:, 12:15], axis=1
            # )
            # ee_orient_error_cumulative = jnp.cumsum(ee_orient_error_running)
            # ee_orient_error_max = jnp.max(ee_orient_error_running)
            # ee_orient_error_total = jnp.sum(ee_orient_error_running)
            # model_errors_max_ee_orient[model_name] = ee_orient_error_max
            # model_errors_total_ee_orient[model_name] = ee_orient_error_total

            # # fill extra evals metrics for current trajectory
            # for state_label in state_labels:
            #     state_idx = state_labels.index(state_label)
            #     state_error = (
            #         y_pred_testing[:, state_idx] - testing_y[:, state_idx]
            #     ) ** 2
            #     extra_eval_metrics[f"{state_label}_rmse"] = (
            #         extra_eval_metrics.get(
            #             f"{state_label}_rmse", jnp.zeros(state_error.shape)
            #         )
            #         + state_error
            #     )
            # base_pos_error = (
            #     jnp.linalg.norm(y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1) ** 2
            # )
            # theta_error = (
            #     jnp.linalg.norm(y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1) ** 2
            # )
            # ee_pos_error = (
            #     jnp.linalg.norm(y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1) ** 2
            # )
            # extra_eval_metrics["base_pos_rmse"] = (
            #     extra_eval_metrics.get("base_pos_rmse", jnp.zeros(base_pos_error.shape))
            #     + base_pos_error
            # )
            # extra_eval_metrics["theta_rmse"] = (
            #     extra_eval_metrics.get(
            #         "theta_rmse", jnp.zeros(base_theta_error_running.shape)
            #     )
            #     + theta_error
            # )
            # extra_eval_metrics["ee_pos_rmse"] = (
            #     extra_eval_metrics.get("ee_pos_rmse", jnp.zeros(ee_pos_error.shape))
            #     + ee_pos_error
            # )
            # ee_orient_error = (
            #     jnp.linalg.norm(
            #         y_pred_testing[:, 12:15] - testing_y[:, 12:15], axis=1
            #     )
            #     ** 2
            # )
            # extra_eval_metrics["ee_orient_rmse"] = (
            #     extra_eval_metrics.get(
            #         "ee_orient_rmse", jnp.zeros(ee_orient_error.shape)
            #     )
            #     + ee_orient_error
            # )

            # # detailed plot of trajectory rollout and errors
            # prepare plots
            fig, axs = plt.subplots(9, 3, figsize=(30, 15))
            # fig_ee_error, axs_ee_error = plt.subplots(8, 1, figsize=(30, 15))
            # fig_base_error, axs_base_error = plt.subplots(4, 2, figsize=(30, 15))

            # plot true data
            for i in range(6):
                axs[i, 0].plot(testing_y[:, i], label="true")
                axs[i, 0].set_title(SPOT_STATE_LABELS[i])

                if i + 6 < testing_y.shape[-1]:
                    axs[i, 1].plot(testing_y[:, i + 6], label="true")
                    axs[i, 1].set_title(SPOT_STATE_LABELS[i + 6])

            for i in range(6):
                if i + 12 < testing_y.shape[-1]:
                    axs[i, 2].plot(testing_y[:, i + 12], label="true")
                    axs[i, 2].set_title(SPOT_STATE_LABELS[i + 12])

            # # prepare error plots
            # axs_ee_error[0].set_title("Running ee position error")
            # axs_ee_error[1].set_title("Running cumulative ee position error")
            # axs_ee_error[2].set_title("Max ee position error")
            # axs_ee_error[3].set_title("Total cumulative ee position error")
            # axs_base_error[0, 0].set_title("Running base position error")
            # axs_base_error[1, 0].set_title("Running cumulative base position error")
            # axs_base_error[2, 0].set_title("Max base position error")
            # axs_base_error[3, 0].set_title("Total cumulative base position error")
            # axs_base_error[0, 1].set_title("Running base theta error")
            # axs_base_error[1, 1].set_title("Running cumulative base theta error")
            # axs_base_error[2, 1].set_title("Max base theta error")
            # axs_base_error[3, 1].set_title("Total cumulative base theta error")
            # axs_ee_error[4].set_title("Running ee orientation error")
            # axs_ee_error[5].set_title("Running cumulative ee orientation error")
            # axs_ee_error[6].set_title("Max ee orientation error")
            # axs_ee_error[7].set_title("Total cumulative ee orientation error")

            # plot simulated trajectory
            for i in range(6):
                axs[i, 0].plot(
                    y_pred_testing[:, i], label=f"pred {model_name}", linestyle="--"
                )

                if i + 6 < testing_y.shape[-1]:
                    axs[i, 1].plot(
                        y_pred_testing[:, i + 6],
                        label=f"pred {model_name}",
                        linestyle="--",
                    )

            for i in range(9):
                if i + 12 < testing_y.shape[-1]:
                    axs[i, 2].plot(
                        y_pred_testing[:, i + 12],
                        label=f"pred {model_name}",
                        linestyle="--",
                    )

            # # plot running and cumulative errors
            # axs_ee_error[0].plot(ee_pos_error_running, label=f"{model_name}")
            # axs_ee_error[1].plot(ee_pos_error_cumulative, label=f"{model_name}")

            # axs_base_error[0, 0].plot(base_pos_error_running, label=f"{model_name}")
            # axs_base_error[1, 0].plot(base_pos_error_cumulative, label=f"{model_name}")

            # axs_base_error[0, 1].plot(base_theta_error_running, label=f"{model_name}")
            # axs_base_error[1, 1].plot(
            #     base_theta_error_cumulative, label=f"{model_name}"
            # )

            # axs_ee_error[4].plot(ee_orient_error_running, label=f"{model_name}")
            # axs_ee_error[5].plot(ee_orient_error_cumulative, label=f"{model_name}")

            # # plot max and total errors
            # axs_ee_error[2].barh(
            #     list(model_errors_max_ee.keys()), list(model_errors_max_ee.values())
            # )
            # axs_ee_error[3].barh(
            #     list(model_errors_total_ee.keys()), list(model_errors_total_ee.values())
            # )
            # axs_base_error[2, 0].barh(
            #     list(model_errors_max_base.keys()), list(model_errors_max_base.values())
            # )
            # axs_base_error[3, 0].barh(
            #     list(model_errors_total_base.keys()),
            #     list(model_errors_total_base.values()),
            # )
            # axs_base_error[2, 1].barh(
            #     list(model_errors_max_base_theta.keys()),
            #     list(model_errors_max_base_theta.values()),
            # )
            # axs_base_error[3, 1].barh(
            #     list(model_errors_total_base_theta.keys()),
            #     list(model_errors_total_base_theta.values()),
            # )

            # axs_ee_error[6].barh(
            #     list(model_errors_max_ee_orient.keys()),
            #     list(model_errors_max_ee_orient.values()),
            # )
            # axs_ee_error[7].barh(
            #     list(model_errors_total_ee_orient.keys()),
            #     list(model_errors_total_ee_orient.values()),
            # )

            # for i in range(6):
            #     axs[i, 0].legend(fontsize=8)
            #     axs[i, 1].legend(fontsize=8)
            #     axs[i, 2].legend(fontsize=8)

            # for i in range(4):
            #     axs_ee_error[i].legend(fontsize=8)
            #     axs_base_error[i, 0].legend(fontsize=8)
            #     axs_base_error[i, 1].legend(fontsize=8)
            #     axs_ee_error[i + 4].legend(fontsize=8)

            if self.wandb_logging:
                wandb.log(
                    {
                        f"sys_id_extra_eval/{traj_id}/plot": wandb.Image(fig),
                        # f"sys_id_extra_eval/{traj_id}/ee_error_plot": wandb.Image(
                        #     fig_ee_error
                        # ),
                        # f"sys_id_extra_eval/{traj_id}/base_error_plot": wandb.Image(
                        #     fig_base_error
                        # ),
                    }
                )

        # # calculate rmse across all eval trajectories
        # for state_label in state_labels:
        #     state_idx = state_labels.index(state_label)
        #     extra_eval_metrics[f"{state_label}_rmse"] = jnp.sqrt(
        #         extra_eval_metrics[f"{state_label}_rmse"] / len(eval_trajectories)
        #     )
        # extra_eval_metrics["base_pos_rmse"] = jnp.sqrt(
        #     extra_eval_metrics["base_pos_rmse"] / len(eval_trajectories)
        # )
        # extra_eval_metrics["theta_rmse"] = jnp.sqrt(
        #     extra_eval_metrics["theta_rmse"] / len(eval_trajectories)
        # )
        # extra_eval_metrics["ee_pos_rmse"] = jnp.sqrt(
        #     extra_eval_metrics["ee_pos_rmse"] / len(eval_trajectories)
        # )

        # extra_eval_metrics["ee_orient_rmse"] = jnp.sqrt(
        #     extra_eval_metrics["ee_orient_rmse"] / len(eval_trajectories)
        # )

        # # log extra eval metrics
        # if self.wandb_logging:
        #     for step in range(step_range):
        #         step_extra_eval_metrics = {
        #             f"sys_id_extra_eval/{state_label}_rmse": float(
        #                 extra_eval_metrics[f"{state_label}_rmse"][step]
        #             )
        #             for state_label in state_labels
        #         }
        #         step_extra_eval_metrics.update(
        #             {
        #                 "sys_id_extra_eval/base_pos_rmse": float(
        #                     extra_eval_metrics["base_pos_rmse"][step]
        #                 ),
        #                 "sys_id_extra_eval/theta_rmse": float(
        #                     extra_eval_metrics["theta_rmse"][step]
        #                 ),
        #                 "sys_id_extra_eval/ee_pos_rmse": float(
        #                     extra_eval_metrics["ee_pos_rmse"][step]
        #                 ),
        #                 "sys_id_extra_eval/step": step,
        #             }
        #         )
        #         step_extra_eval_metrics.update(
        #             {
        #                 "sys_id_extra_eval/ee_orient_rmse": float(
        #                     extra_eval_metrics["ee_orient_rmse"][step]
        #                 )
        #             }
        #         )
        #         wandb.log(step_extra_eval_metrics)


if __name__ == "__main__":
    # TODO: add test
    print("Test execution not implemented.")
    pass
