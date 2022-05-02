import jax
import jax.numpy as jnp
import optax

from typing import List, Optional, Callable, Tuple, Dict
from collections import OrderedDict
from sim_transfer.models.abstract_model import AbstactRegressionModel
from sim_transfer.modules import BatchedMLP

from functools import partial
from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

class BNN_SVGD(AbstactRegressionModel):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey,
                 likelihood_std: float = 0.2,
                 num_particles: int = 10,
                 bandwidth: float = 10.0,
                 data_batch_size: int = 16,
                 num_steps: int = 10000,
                 lr = 1e-3,
                 normalize_data: bool = True,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         normalize_data=normalize_data)
        self.likelihood_std = likelihood_std * jnp.ones(output_size)
        self.data_batch_size = data_batch_size
        self.num_steps = num_steps
        self.num_particles = num_particles
        self.bandwidth = bandwidth

        # initialize batched NN
        self.batched_model = BatchedMLP(input_size=input_size, output_size=output_size,
                                        num_batched_modules=num_particles,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        hidden_activation=hidden_activation,
                                        last_activation=last_activation,
                                        rng_key=self.rng_key)
        self.params_stack = self.batched_model.param_vectors_stacked

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

        # initialize kernel
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, particles: jnp.ndarray):
        particles_copy = jax.lax.stop_gradient(particles)
        k = self.kernel.matrix(particles, particles_copy)
        return jnp.sum(k), k

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array):
        # SVGD updates
        nll, grad_q = jax.value_and_grad(self._nll)(param_vec_stack, x_batch, y_batch)
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(param_vec_stack)
        grad = k @ grad_q + grad_k / self.num_particles

        updates, opt_state = self.optim.update(grad, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)

        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(train_nll_loss=nll, avg_grad_q=jnp.mean(grad_q), avg_grad_k=jnp.mean(grad_q),
                            avg_triu_k=avg_triu_k)

        return opt_state, param_vec_stack, stats

    def _nll(self, param_vec_stack: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        pred_raw = self.batched_model.forward_vec(x_batch, param_vec_stack)
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw, likelihood_std).log_prob(y_batch)
        return - jnp.mean(log_prob)

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True) -> tfd.Distribution:
        self.batched_model.param_vectors_stacked = self.params_stack
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        pred_dist = self._to_pred_dist(y_pred_raw, likelihood_std=self.likelihood_std, include_noise=include_noise)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_dist = self.predict_dist(x=x, include_noise=include_noise)
        return pred_dist.mean, pred_dist.stddev

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred


if __name__ == '__main__':
    def key_iter():
        key = jax.random.PRNGKey(7644)
        while True:
            key, new_key = jax.random.split(key)
            yield new_key
    key_iter = key_iter()

    fun = lambda x: 2 * x + 2 * jnp.sin(2 * x)

    num_train_points = 10
    x_train = jax.random.uniform(next(key_iter), shape=(num_train_points, 1), minval=-5, maxval=5)
    y_train = fun(x_train) + 0.1 * jax.random.normal(next(key_iter), shape=x_train.shape)

    num_test_points = 100
    x_test = jax.random.uniform(next(key_iter), shape=(num_test_points, 1), minval=-5, maxval=5)
    y_test = fun(x_test) + 0.1 * jax.random.normal(next(key_iter), shape=x_test.shape)


    bnn = BNN_SVGD(1, 1, next(key_iter), num_steps=20000, bandwidth=10.)
    #bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=20000)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        bnn.plot_1d(x_train, y_train, true_fun=fun, title=f'iter {(i+1) * 2000}')

