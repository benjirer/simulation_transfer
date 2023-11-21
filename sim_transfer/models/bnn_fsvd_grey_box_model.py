from sim_transfer.models.bnn_fsvgd import BNN_FSVGD
import jax.numpy as jnp
from typing import Optional, Dict, Union, Tuple
from collections import OrderedDict
import jax
from typing import NamedTuple
from sim_transfer.modules.distribution import AffineTransform
from sim_transfer.sims.simulators import FunctionSimulator
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial
import time
import copy
from sim_transfer.modules.util import aggregate_stats
import wandb
import numpy as np


class BNN_FSVGD_GreyBox(BNN_FSVGD):

    def __init__(self, sim: FunctionSimulator, lr_sim: float = None, weight_decay_sim: float = 0.0,
                 num_sim_model_train_steps: int = 10_000,
                 *args, **kwargs):
        super().__init__(domain=sim.domain, *args, **kwargs)
        self.sim = sim
        param_key = self._next_rng_key()
        self.num_sim_model_train_steps = num_sim_model_train_steps
        sim_params = self.sim.sample_params(param_key)
        likelihood_std = -1. * jnp.ones(self.output_size)
        self.params_sim = {'sim_params': sim_params, 'likelihood_std': likelihood_std}
        self.optim_sim = None
        self._y_mean_sim, self._y_std_sim = jnp.copy(self._y_mean), jnp.copy(self._y_std)
        if lr_sim:
            self.lr_sim = lr_sim
        else:
            self.lr_sim = self.lr
        if weight_decay_sim is not None:
            self.weight_decay_sim = weight_decay_sim
        else:
            self.weight_decay_sim = self.weight_decay
        self._init_optim()
        self._init_sim_optim()

    def reinit(self, rng_key: Optional[jax.random.PRNGKey] = None):
        """ Reinitializes the model parameters and the optimizer state."""
        if rng_key is None:
            key_rng = self._next_rng_key()
            key_model = self._next_rng_key()
            param_key = self._next_rng_key()
        else:
            key_model, key_rng, param_key = jax.random.split(rng_key, 3)
        self._rng_key = key_rng  # reinitialize rng_key
        self.batched_model.reinit_params(key_model)  # reinitialize model parameters
        self.params['nn_params_stacked'] = self.batched_model.param_vectors_stacked
        sim_params = self.sim.sample_params(param_key)
        likelihood_std = -1. * jnp.ones(self.output_size)
        self.params_sim = {'sim_params': sim_params, 'likelihood_std': likelihood_std}
        self._init_likelihood()  # reinitialize likelihood std
        self._init_optim()  # reinitialize optimizer
        self._init_sim_optim()

    def _init_sim_optim(self):
        """ Initializes the optimizer and the optimizer state.
        Sets the attributes self.optim and self.opt_state. """
        if self.weight_decay_sim > 0:
            self.optim_sim = optax.adamw(learning_rate=self.lr_sim, weight_decay=self.weight_decay_sim)
        else:
            self.optim_sim = optax.adam(learning_rate=self.lr_sim)
        self.opt_state_sim = self.optim_sim.init(self.params_sim)

    def _preprocess_train_data_sim(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Convert data to float32, ensure 2d shape and normalize if necessary"""
        if self.normalize_data:
            if self._need_to_compute_norm_stats:
                self._compute_normalization_stats_sim(x_train, y_train)
            x_train, y_train = self._normalize_data_sim(x_train, y_train)
        else:
            x_train, y_train = self._ensure_atleast_2d_float(x_train, y_train)
        return x_train, y_train

    def _compute_normalization_stats_sim(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        # computes the empirical normalization stats and stores as private variables
        x, y = self._ensure_atleast_2d_float(x, y)
        self._x_mean = jnp.mean(x, axis=0)
        self._y_mean_sim = jnp.mean(y, axis=0)
        self._x_std = jnp.std(x, axis=0)
        self._y_std_sim = jnp.std(y, axis=0)

    def _normalize_data_sim(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        # normalized the given data with the normalization stats
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_normalized = (x - self._x_mean[None, :]) / (self._x_std[None, :] + eps)
        assert x_normalized.shape == x.shape
        if y is None:
            return x_normalized
        else:
            y_normalized = self._normalize_y_sim(y, eps=eps)
            assert y_normalized.shape == y.shape
            return x_normalized, y_normalized

    def _unnormalize_data_sim(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_unnorm = x * (self._x_std[None, :] + eps) + self._x_mean[None, :]
        assert x_unnorm.shape == x.shape
        if y is None:
            return x_unnorm
        else:
            y_unnorm = self._unnormalize_y_sim(y, eps=eps)
            return x_unnorm, y_unnorm

    def _normalize_y_sim(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_normalized = (y - self._y_mean_sim) / (self._y_std_sim + eps)
        assert y_normalized.shape == y.shape
        return y_normalized

    def _unnormalize_y_sim(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_unnorm = y * (self._y_std_sim + eps) + self._y_mean_sim
        assert y_unnorm.shape == y.shape
        return y_unnorm

    def _sim_step(self, opt_state_sim: optax.OptState, params_sim: Dict,
                  x_batch: jnp.array, y_batch: jnp.array, num_train_points: Union[float, int]):
        loss, grad = jax.value_and_grad(
            self._sim_loss)(
            params_sim,
            x_batch, y_batch,
            num_train_points)
        updates, new_opt_state_sim = self.optim_sim.update(grad, opt_state_sim, params_sim)
        new_params_sim = optax.apply_updates(params_sim, updates)
        stats = OrderedDict(sim_params_nll=loss)
        return new_opt_state_sim, new_params_sim, stats

    def _step_grey_box(self, opt_state_sim: optax.OptState, params_sim: Dict,
                       x_batch: jnp.array, y_batch: jnp.array,
                       num_train_points: Union[float, int]):
        new_opt_state_sim, new_params_sim, stats = self._sim_step(
            opt_state_sim, params_sim, x_batch, y_batch, num_train_points
        )
        return new_opt_state_sim, new_params_sim, stats

    def sim_model_step(self, x: jnp.array, params_sim: NamedTuple, normalized_x: bool = False,
                       normalized_y: bool = False):
        if normalized_x:
            x = self._unnormalize_data_sim(x)
        y = self.sim.evaluate_sim(x, params_sim)
        if normalized_y:
            y_normalized = self._normalize_y_sim(y)
            assert y_normalized.shape == y.shape
            y = y_normalized
        return y

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True):
        y_pred_raw = self.normalized_batched_predictions(x)
        pred_dist = self._to_pred_dist(y_pred_raw, likelihood_std=self.likelihood_std, include_noise=include_noise)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def normalized_batched_predictions(self, x: jnp.ndarray):
        # sim model takes unormalized input and predicts unnormalized output.
        y_sim_raw = self.sim_model_step(x, params_sim=self.params_sim['sim_params'], normalized_x=False,
                                        normalized_y=False)
        # We scale the output of the sim model with y_std to obtain y_sim_raw
        y_sim_raw = self._ensure_atleast_2d_float(y_sim_raw)
        assert y_sim_raw.shape[-1] == self.output_size
        y_sim_scaled = y_sim_raw / (self._y_std + 1e-8)
        # normalize data to get normalized output of the neural network.
        x = self._normalize_data(x)
        self.batched_model.param_vectors_stacked = self.params['nn_params_stacked']
        y_pred_raw = self.batched_model(x)
        # total normalized output is output of NNs + y_sim_scaled. To unnormalize we get (y_nn + y_sim_scaled) * std
        # + mean -> y_nn * std + mean + y_sim_scaled * std -> y_nn_unnormalized + y_sim_raw
        y_pred_raw = jax.vmap(lambda z: z + y_sim_scaled)(y_pred_raw)
        return y_pred_raw

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        sim_model_prediction = self.sim_model_step(x, self.params_sim['sim_params'], normalized_x=False,
                                                   normalized_y=False)
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        y_pred = jax.tree_util.tree_map(lambda y: y + sim_model_prediction, y_pred)
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred

    def _sim_loss(self, params_sim: Dict, x_batch: jnp.array, y_batch: jnp.array,
                  num_train_points: Union[float, int]):

        normalized_sim_model_prediction = self.sim_model_step(x_batch, params_sim['sim_params'], normalized_x=True,
                                                              normalized_y=True)

        # get likelihood std
        likelihood_std = jnp.exp(params_sim['likelihood_std']) if self.learn_likelihood_std else self.likelihood_std

        def _ll(pred, y):
            return tfd.MultivariateNormalDiag(pred, likelihood_std).log_prob(y)

        ll = jax.vmap(_ll)
        nll = - num_train_points * self.likelihood_exponent * jnp.mean(
            ll(normalized_sim_model_prediction, y_batch), axis=0)
        return nll

    @partial(jax.jit, static_argnums=(0,))
    def _step_grey_box_jit(self, *args, **kwargs):
        return self._step_grey_box(*args, **kwargs)

    def step_grey_box(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]):
        self.opt_state_sim, self.params_sim, stats = self._step_grey_box_jit(
            opt_state_sim=self.opt_state_sim, params_sim=self.params_sim, x_batch=x_batch,
            y_batch=y_batch, num_train_points=num_train_points)
        return stats

    def fit_sim_prior(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
                      y_eval: Optional[jnp.ndarray] = None,
                      num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
                      log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss',
                      keep_the_best: bool = False,
                      per_dim_metrics: bool = False):

        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]

        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data_sim(x_train, y_train)
        batch_size = min(self.data_batch_size, x_train.shape[0])
        train_loader = self._create_data_loader(x_train, y_train, batch_size=batch_size)
        num_train_points = x_train.shape[0]

        num_sim_model_train_steps = self.num_sim_model_train_steps if num_sim_model_train_steps is None else \
            num_sim_model_train_steps
        samples_cum_period = 0.0
        stats_list = []
        t_start_period = time.time()

        # do callback before training loop starts in case the specific method wants to do something for
        # which it needs the normalization stats of the data
        self._before_training_loop_callback()

        best_objective = jnp.array(jnp.inf)
        best_params = None

        # training loop
        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            samples_cum_period += x_batch.shape[0]

            # perform the train step

            stats = self.step_grey_box(x_batch, y_batch, num_train_points)
            stats_list.append(stats)

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_sample_ms = duration_sec / samples_cum_period * 1000
                stats_agg = aggregate_stats(stats_list)

                if evaluate:
                    eval_stats = self.eval_sim(x_eval, y_eval, prefix='eval_', per_dim_metrics=per_dim_metrics)
                    if keep_the_best:
                        if metrics_objective in eval_stats.keys():
                            if eval_stats[metrics_objective] < best_objective:
                                best_objective = eval_stats[metrics_objective]
                                best_params = copy.deepcopy(self.params_sim)
                        else:
                            if stats['sim_params_nll'] < best_objective:
                                best_objective = stats['sim_params_nll']
                                best_params = copy.deepcopy(self.params_sim)
                    stats_agg.update(eval_stats)
                if log_to_wandb:
                    log_dict = {f'regression_model_training/{n}': float(v) for n, v in stats_agg.items()}
                    wandb.log(log_dict | {'x_axis/bnn_step': step})
                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_sim_model_train_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                stats_list = []
                samples_cum_period = 0
                t_start_period = time.time()

            if step >= num_sim_model_train_steps:
                break

        if keep_the_best and best_params is not None:
            self.params_sim = best_params
            print(f'Keeping the best model with {metrics_objective}={best_objective:.4f}')
            if log_to_wandb:
                wandb.log({metrics_objective: best_objective})

    def eval_sim(self, x: jnp.ndarray, y: np.ndarray, prefix: str = '', per_dim_metrics: bool = False) \
            -> Dict[str, jnp.ndarray]:
        """ Evaluates the model on the given data and returns a dictionary of evaluation metrics.
        Args:
            x: inputs
            y: targets
            prefix: (str) prefix for the returned dictionary keys
            per_dim_metrics: (bool) whether to also return the per-dimension MAE

        Returns: dict of evaluation metrics
        """
        # make predictions
        x, y = self._ensure_atleast_2d_float(x, y)
        pred_y = self.sim_model_step(x, self.params_sim['sim_params'])

        rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_y - y) ** 2, axis=-1)))
        eval_stats = {'rmse': rmse}

        # compute per-dimension MAE
        if per_dim_metrics:
            mae_per_dim = jnp.mean(jnp.abs(pred_y - y), axis=0)
            rmse_per_dim = jnp.sqrt(jnp.mean((pred_y - y) ** 2, axis=0))
            eval_stats.update({f'per_dim_metrics/mae_{i}': mae_per_dim[i] for i in range(self.output_size)})
            eval_stats.update({f'per_dim_rmse/rmse_{i}': rmse_per_dim[i] for i in range(self.output_size)})

        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        # make sure that all stats are python native floats
        eval_stats = {name: float(val) for name, val in eval_stats.items()}
        return eval_stats

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
            y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None,
            num_sim_model_train_steps: Optional[int] = None, log_period: int = 1000,
            log_to_wandb: bool = False, metrics_objective: str = 'train_nll_loss', keep_the_best: bool = False,
            per_dim_metrics: bool = False):

        self.fit_sim_prior(x_train, y_train, x_eval, y_eval, num_sim_model_train_steps, log_period,
                           log_to_wandb, metrics_objective, keep_the_best, per_dim_metrics)
        y_train = y_train - self.sim_model_step(x_train, self.params_sim['sim_params'])
        super().fit(
            x_train, y_train, x_eval, y_eval, num_steps, log_period, log_to_wandb,
            metrics_objective,
            keep_the_best, per_dim_metrics
        )


if __name__ == '__main__':
    from sim_transfer.sims.simulators import RaceCarSim

    sim = RaceCarSim(encode_angle=True, use_blend=True)


    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key


    key_iter = key_iter()
    NUM_DIM_X = 9
    NUM_DIM_Y = 7
    obs_noise_std = 1e-3
    weight_decay = 1e-5
    x_train, y_train, x_test, y_test = sim.sample_datasets(rng_key=next(key_iter), num_samples_train=20000,
                                                           obs_noise_std=obs_noise_std, param_mode='typical')
    bnn = BNN_FSVGD_GreyBox(sim=sim, input_size=NUM_DIM_X, output_size=NUM_DIM_Y, rng_key=next(key_iter),
                            num_train_steps=20000,
                            bandwidth_svgd=1.0, likelihood_std=obs_noise_std, likelihood_exponent=1.0,
                            normalize_likelihood_std=True, learn_likelihood_std=False, weight_decay=weight_decay)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000, per_dim_metrics=True)
