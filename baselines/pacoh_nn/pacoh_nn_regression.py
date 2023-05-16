import time
import tensorflow as tf
import tensorflow_probability as tfp

from baselines.pacoh_nn import modules
from baselines.pacoh_nn.modules.neural_network import BatchedFullyConnectedNN
from baselines.pacoh_nn.modules.likelihood import GaussianLikelihood
from baselines.pacoh_nn.modules.prior_posterior import BatchedGaussianPrior
from baselines.pacoh_nn.modules.affine_transform import AffineTransform
from baselines.pacoh_nn.bnn.bnn_svgd import BayesianNeuralNetworkSVGD
from baselines.pacoh_nn.modules import MetaDatasetSampler
from baselines.pacoh_nn.meta_algo import MetaLearner

class PACOH_NN_Regression(MetaLearner):

    def __init__(self, meta_train_data, lr=2e-3,  hidden_layer_sizes=(32, 32, 32, 32), activation='relu',
                 learn_likelihood=True, likelihood_std=0.1, meta_batch_size=4, batch_size=5, num_iter_meta_train=30000,
                 num_iter_meta_test=5000, n_samples_per_prior=10, num_hyper_posterior_particles=3,
                 num_posterior_particles=5, prior_weight=0.1, hyper_prior_weight=1e-4, hyper_prior_nn_std=0.4,
                 hyper_prior_log_var_mean=-3.0, hyper_prior_likelihood_log_var_mean_mean=-8,
                 hyper_prior_likelihood_log_var_log_var_mean=-4.,
                 bandwidth=10.0, random_seed=None):

        super().__init__(random_seed=random_seed)

        self.hyper_prior_weight = hyper_prior_weight

        # data handling & settings
        self.n_batched_models_train = num_hyper_posterior_particles * n_samples_per_prior
        self.n_batched_models_test = num_hyper_posterior_particles * num_posterior_particles
        self.n_samples_per_prior = n_samples_per_prior
        self.n_batched_priors = num_hyper_posterior_particles

        # prepare config for meta eval model
        self.eval_model_config = {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation,
                                  'learn_likelihood': learn_likelihood,
                                  'n_particles': num_posterior_particles * self.n_batched_priors,
                                  'prior_weight': prior_weight, 'bandwidth': bandwidth,
                                  'batch_size': batch_size}
        self.num_iter_meta_test = num_iter_meta_test

        self._process_meta_train_data(meta_train_data, meta_batch_size=meta_batch_size, batch_size=batch_size,
                                      n_batched_models_train=self.n_batched_models_train)

        self.mll_pre_factor = self._compute_mll_prefactor()

        # meta-training settings
        self.num_iter_meta_train = num_iter_meta_train

        # meta-testing / evaluation settings
        self.process_eval_batch = self.meta_train_sampler.tasks[0].process_eval_batch
        self.posterior_inference_batch_size = batch_size

        """ Setup and initialize model and other relevant components """

        # setup NN
        self.nn_model = BatchedFullyConnectedNN(self.n_batched_models_train, self.output_dim, hidden_layer_sizes,
                                                activation=activation)
        self.nn_model.build((None, self.input_dim))

        # setup likelihood
        self.likelihood = GaussianLikelihood(self.output_dim, self.n_batched_models_train,
                                             std=likelihood_std, trainable=learn_likelihood)
        self.learn_likelihood = learn_likelihood
        self.likelihood_std = likelihood_std

        # setup prior
        self.nn_param_size = self.nn_model.get_variables_stacked_per_model().shape[-1]
        self.likelihood_param_size = self.output_dim if learn_likelihood else 0

        self.prior_module = BatchedGaussianPrior(batched_model=self.nn_model, n_batched_priors=self.n_batched_priors,
                                                 likelihood_param_size=self.likelihood_param_size,
                                                 likelihood_prior_mean=tf.math.log(0.01),
                                                 likelihood_prior_std=1.0,
                                                 name='gaussian_prior')

        self.hyper_prior_module = modules.GaussianHyperPrior(self.prior_module,
                                                             mean_mean=0.0, bias_mean_std=hyper_prior_nn_std,
                                                             kernel_mean_std=hyper_prior_nn_std,
                                                             log_var_mean=hyper_prior_log_var_mean, bias_log_var_std=hyper_prior_nn_std,
                                                             kernel_log_var_std=hyper_prior_nn_std,
                                                             likelihood_log_var_mean_mean=hyper_prior_likelihood_log_var_mean_mean,
                                                             likelihood_log_var_mean_std=1.0,
                                                             likelihood_log_var_log_var_mean=hyper_prior_likelihood_log_var_log_var_mean,
                                                             likelihood_log_var_log_var_std=0.2)

        self.hyper_posterior_particles = tf.Variable(tf.expand_dims(self.prior_module.get_variables_stacked_per_model(), axis=0))

        self.affine_transform = AffineTransform(normalization_mean=self.y_mean, normalization_std=self.y_std)
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=bandwidth)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def meta_fit(self, meta_val_data=None, log_period=500, eval_period=1000, plot_prior_during_training=False,
                 plot_period=2000):
        """
        fits the hyper-posterior (PACOH) particles with SVGD

        Args:
            meta_val_data: list of valid tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]
            log_period (int):  number of steps after which to print the meta-train loss
            eval_period (int): number of steps after which to perform meta-testing and print the evaluation stats
            plot_prior_during_training (bool): whether to plot the prior during training
                                                (only supported if input_dim == output_dim == 1)
            plot_period (int): number of steps after which to plot the prior
        """
        assert not plot_prior_during_training or (self.input_dim == self.output_dim == 1), \
            "plotting the prior is only supported if input_dim == output_dim == 1"

        print("Start meta-training -------------------- ")
        t = time.time()
        for iter in range(self.num_iter_meta_train):

            meta_batch_x, meta_batch_y, n_train_samples, _, _ = self.meta_train_sampler.get_meta_batch()
            log_prob = self.step(meta_batch_x, meta_batch_y, n_train_samples)

            message = ''
            # print training loss
            if plot_prior_during_training and iter % plot_period == 0:
                self.plot_prior(plot_pred_lines=True, plot_data=True, max_task_to_plot=10,
                                title='prior samples @ iter %i' % iter, show=True)

            if iter % log_period == 0 or iter % eval_period == 0:
                avg_log_prob = - tf.reduce_mean(log_prob).numpy()
                message += '\nIter %d/%d - Time %.2f sec' % (iter, self.num_iter_meta_train, time.time() - t)
                message += ' - Train-Loss: %.5f' % avg_log_prob

                # run validation and print results
                if meta_val_data is not None and iter % eval_period == 0 and iter > 0:
                    eval_metrics_mean, eval_metrics_std = self.meta_eval_datasets(meta_val_data)
                    for key in eval_metrics_mean:
                        message += '- Val-%s: %.3f +- %.3f' % (key, eval_metrics_mean[key], eval_metrics_std[key])

                t = time.time()

            if len(message) > 0:
                print(message)

        loss = - tf.reduce_mean(log_prob).numpy()
        return loss

    def meta_eval_datasets(self, meta_valid_data, max_tasks_parallel=5):
        """
        meta-testing functionality - Runs posterior inference for the tasks in meta_valid_data
        and reports evaluation metrics. For the posterior inference the context data is used while the evaluation
        metrics are computed on the left-out test sets

        Args:
            meta_valid_data: list of valid tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]
            max_tasks_parallel (int): maximum number of tasks to evaluate on in parallel. If
                                      max_tasks_parallel < n_test_tasks it will perform meta-testing in batches

        Returns: (eval_metrics_mean, eval_metrics_std) each a dict of evaluation results
        """
        meta_valid_sampler = MetaDatasetSampler(meta_valid_data, self.posterior_inference_batch_size,
                                                n_batched_models=self.n_batched_models_test, tiled=False)
        meta_valid_sampler.copy_standardization_stats(self.meta_train_sampler)

        eval_tasks_batches = self._split_into_batches(meta_valid_sampler.tasks, max_batch_size=max_tasks_parallel)

        # Start training the BNN models
        print("\tStart meta-test posterior inference in %i batches ------------------"%len(eval_tasks_batches))

        eval_metrics_dict_per_task = []
        for eval_batch_id, task_batch in enumerate(eval_tasks_batches):
            print("\tMeta-Test batch #%i consisting of %i tasks----"%(eval_batch_id+1, len(task_batch)))
            # initialize evaluation models and define their training step
            eval_models, eval_models_step = self._setup_meta_test_models_and_step(meta_test_tasks=task_batch)
            # perform training and evaluation
            self._meta_test_training_loop(task_batch, eval_models, eval_models_step, log_period=10000,
                                          num_iter=self.num_iter_meta_test, eval_period=10000)
            _, _, eval_metrics_grouped = self._meta_test_models_eval(task_batch, eval_models)
            eval_metrics_dict_per_task.append(eval_metrics_grouped)

        eval_metrics_mean, eval_metrics_std = self._aggregate_eval_metrics_across_tasks(eval_metrics_dict_per_task)
        return eval_metrics_mean, eval_metrics_std

    def meta_predict(self, x_context, y_context, x_test):
        """
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y)

        Args:
          context_x: (ndarray) context input data for which to compute the posterior
          context_y: (ndarray) context targets for which to compute the posterior
          test_x: (ndarray) query input data of shape (n_samples, ndim_x)

        Returns:
          (y_pred, pred_dist) predicted means corresponding to the posterior particles and aggregate predictive
                              p(y|test_x, test_context_x, context_y)
        """
        normalization_stats_dict = self._get_normalization_stats_dict()
        eval_model = BayesianNeuralNetworkSVGD(x_context, y_context, **self.eval_model_config,
                                               meta_learned_prior=self.prior_module,
                                               normalization_stats=normalization_stats_dict)
        eval_model.fit(num_iter_fit=self.num_iter_meta_test)
        y_pred, pred_dist = eval_model.predict(x_test)

        return y_pred, pred_dist

    def meta_eval(self, x_context, y_context, x_test, y_test):
        """
        computes the average test log likelihood, rmse and calibration error n test data

        Args:
          context_x: (ndarray) context input data for which to compute the posterior
          context_y: (ndarray) context targets for which to compute the posterior
          test_x: (ndarray) test input data of shape (n_samples, ndim_x)
          test_y: (ndarray) test target data of shape (n_samples, ndim_y)

        Returns: dict containing the average the test log likelihood, the rmse and, if ndim_y = 1, the calibration error
        """
        y_pred, pred_dist = self.meta_predict(x_context, y_context, x_test)
        return self.likelihood.calculate_eval_metrics(pred_dist, y_test)

    def plot_prior(self, x_min=None, x_max=None, ax=None, plot_pred_std=False, plot_pred_lines=False,
                   plot_data=False, max_task_to_plot=None, title=None, show=False):
        """
        Samples function from the (meta-learned) prior and plots them (only for 1d to 1d meta_train_data)
        Args:
            x_min (float): left plot limit
            x_max (float): left plot limit
            ax (pyplot.Axis): pyplot Axis object on which to plot
            plot_pred_std (bool): whether to plot the stddev
            plot_pred_lines (bool): whether to plot indivitual functions sampled from the prior
            plot_data (bool): whether to plot the meta-train data
            max_task_to_plot (int): how many tasks to plot
            title (str): plot title
            show (bool): whether to show the plot
        """
        assert self.input_dim == self.input_dim == 1, 'plotting only supported for 1d to 1d function'
        from matplotlib import pyplot as plt
        if x_min is None:
            x_min = tf.reduce_min([task.x_min for task in self.meta_train_sampler.tasks])
        if x_max is None:
            x_max = tf.reduce_max([task.x_max for task in self.meta_train_sampler.tasks])

        x_flat = tf.linspace(x_min, x_max, 100)
        x = tf.reshape(x_flat, (100, 1))
        y_pred, pred_dist = self._predict(x, sample_functions=True, sample_from_prior=True)

        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))

        if plot_pred_std:
            std = tf.reshape(pred_dist.stddev(), (1, -1))[0]
            mean = tf.reshape(pred_dist.mean(), (1, -1))[0]
            ax.fill_between(x_flat, mean - std, mean + std, alpha=0.2)

        if plot_pred_lines:
            for i in range(y_pred.shape[0]):
                ax.plot(x_flat, tf.reshape(y_pred[i], (1, -1))[0], color='green', alpha=0.3, linewidth=1)

        if plot_data:
            for i, task in enumerate(self.meta_train_sampler.tasks):
                if i < max_task_to_plot:
                    x_train, y_train = task.train_data
                    plt.scatter(x_train, y_train, s=5, alpha=0.4)
                else:
                    break

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

    def plot_posterior(self, x_context, y_context, x_plot, ax=None, plot_context_data=True, show=False):
        """
        plots the predictive distribution of a BNN trained with the context data and the meta-learned prior

        Args:
          context_x: (ndarray) context input data for which to compute the posterior
          context_y: (ndarray) context targets for which to compute the posterior
          x_plot: (ndarray) points at which to plot the data
          ax (pyplot.Axis): pyplot Axis object on which to plot
        """
        from matplotlib import pyplot as plt
        assert self.input_dim == 1 and self.output_dim == 1
        y_pred, pred_dist = self.meta_predict(x_context, y_context, x_plot)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # plot predictive mean and confidence interval
        ax.plot(x_plot, pred_dist.mean())
        lcb, ucb = pred_dist.mean() - 2 * pred_dist.stddev(), pred_dist.mean() + 2 * pred_dist.stddev()
        ax.fill_between(x_plot, lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.2)

        for i in range(y_pred.shape[0]):
            ax.plot(x_plot, y_pred[i], color='green', alpha=0.2, linewidth=1.0)

        if plot_context_data:
            ax.scatter(x_context, y_context)

        if show:
            plt.show()

    @tf.function
    def step(self, meta_batch_x, meta_batch_y, n_train_samples):
        """
        performs one meta-training optimization step (SVGD step on the hyper-posterior)
        """
        # compute score (grads of log_prob)
        log_prob, grads_log_prob = self._pacoh_log_prob_and_grad(meta_batch_x, meta_batch_y, n_train_samples)
        score = grads_log_prob[0]

        # SVGD
        particles = self.hyper_posterior_particles
        K_XX, grad_K_XX = self._get_kernel_matrix_and_grad(particles)
        svgd_grads = - (tf.matmul(K_XX, score) - grad_K_XX) / particles.shape[1]

        self.optimizer.apply_gradients([(svgd_grads, self.hyper_posterior_particles)])
        return log_prob

    def _sample_params_from_prior(self):
        self.prior_module.set_variables_vectorized(self.hyper_posterior_particles)

        param_sample = self.prior_module.sample(self.n_samples_per_prior)
        param_sample = tf.reshape(param_sample, (self.n_batched_models_train, param_sample.shape[-1]))
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(param_sample)
        return nn_params, likelihood_std

    def _pacoh_log_prob_and_grad(self, meta_batch_x, meta_batch_y, n_train_samples):
        with tf.GradientTape() as tape:
            tape.watch(self.hyper_posterior_particles)
            log_likelihood = self._estimate_mll(self.hyper_posterior_particles, meta_batch_x, meta_batch_y, n_train_samples)
            log_prior_prob = self.hyper_prior_module.log_prob_vectorized(self.hyper_posterior_particles,
                                                                         model_params_prior_weight=self.hyper_prior_weight)
            log_prob = log_likelihood + log_prior_prob

        grads = tape.gradient(log_prob, self.hyper_posterior_particles)
        return log_prob, grads

    def _estimate_mll(self, prior_samples, meta_batch_x, meta_batch_y, n_train_samples):
        param_sample = self.prior_module.sample_parametrized(self.n_samples_per_prior, prior_samples)
        log_likelihood = self._compute_likelihood_across_tasks(param_sample, meta_batch_x, meta_batch_y)

        # multiply by sqrt of m and apply logsumexp
        neg_log_likelihood = log_likelihood * tf.math.sqrt(n_train_samples[:, None, None])
        mll = tf.math.reduce_logsumexp(neg_log_likelihood, axis=-1) - tf.math.log(
            tf.cast(self.n_samples_per_prior, dtype=tf.float32))

        # sum over meta-batch size and adjust for number of tasks
        mll_sum = tf.reduce_sum(mll, axis=0) * (self.meta_train_sampler.n_tasks / self.meta_train_sampler.meta_batch_size)

        return self.mll_pre_factor * mll_sum

    def _compute_likelihood_across_tasks(self, params, meta_batch_x, meta_batch_y):
        """ compute the average likelihood, i.e. the mean of the likelihood for the points in the batch (x,y)
            If you want an unbiased estimator of the dataset likelihood, set the prefactor to the number of points

            Returns: log_likelihood_across_tasks with shape (meta_batch_size, n_hyper_posterior_samples, n_prior_samples)
                """
        params = tf.reshape(params, (self.n_batched_models_train, params.shape[-1]))
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(params)

        # iterate over tasks
        log_likelihood_list = []
        for i in range(self.meta_train_sampler.meta_batch_size):
            x = meta_batch_x[i]
            y = meta_batch_y[i]

            # NN forward pass
            y_hat = self.nn_model.call_parametrized(x, nn_params)

            # compute likelihood
            log_likelihood = self.likelihood.log_prob(y_hat, y, likelihood_std)
            log_likelihood = tf.reshape(log_likelihood, (self.n_batched_priors, self.n_samples_per_prior))
            log_likelihood_list.append(log_likelihood)

        log_likelihood_across_tasks = tf.stack(log_likelihood_list)
        return log_likelihood_across_tasks

    def _get_kernel_matrix_and_grad(self, X):
        X2 = tf.identity(X)
        with tf.GradientTape() as tape:
            tape.watch(X)
            K_XX = self.kernel.matrix(X, X2)
        K_grad = tape.gradient(K_XX, X)
        return K_XX, K_grad

    def _setup_meta_test_models_and_step(self, meta_test_tasks):
        # initialize BNN models with the meta-learned prior for each meta-test task
        meta_learned_prior = self.prior_module
        eval_models = []
        for meta_test_task in meta_test_tasks:
            # each meta_test_task corresponds to a dataset sampler
            eval_model = BayesianNeuralNetworkSVGD(*meta_test_task.train_data, **self.eval_model_config,
                                                   meta_learned_prior=self.prior_module,
                                                   normalization_stats=self._get_normalization_stats_dict())

            eval_models.append(eval_model)

        # define parallel training step on all the models
        @tf.function
        def eval_models_step_tf(batch_per_model):
            log_likelihood_per_eval_model = tf.stack([tf.reduce_mean(eval_model.step(batch[0], batch[1]))
                                                      for eval_model, batch in zip(eval_models, batch_per_model)])

            return log_likelihood_per_eval_model

        def eval_models_step():
            # prepare data
            batch_per_eval_model = [task.get_batch() for task in meta_test_tasks]

            # run step in parallel
            log_likelihood_per_eval_model = eval_models_step_tf(batch_per_eval_model)

            return log_likelihood_per_eval_model

        assert len(eval_models) == len(meta_test_tasks)
        return eval_models, eval_models_step

    def _compute_mll_prefactor(self):
        dataset_sizes = tf.convert_to_tensor(self.meta_train_sampler.n_train_samples, dtype=tf.float32)
        harmonic_mean_dataset_size = 1. / (tf.reduce_mean(1. / dataset_sizes))
        return 1 / ((tf.math.sqrt(harmonic_mean_dataset_size * self.meta_train_sampler.n_tasks) + 1))

    def _split_into_nn_params_and_likelihood_std(self, params):
        tf.assert_equal(tf.rank(params), 2)
        tf.assert_equal(params.shape[-1], self.nn_param_size + self.likelihood_param_size)
        n_particles = params.shape[0]
        nn_params = params[:, :self.nn_param_size]
        if self.likelihood_param_size > 0:
            likelihood_std = tf.exp(params[:, -self.likelihood_param_size:])
        else:
            likelihood_std = tf.ones((n_particles, self.output_dim)) * self.likelihood_std

        tf.assert_equal(likelihood_std.shape, (n_particles, self.output_dim))
        return nn_params, likelihood_std

    def _process_meta_train_data(self, meta_train_data, meta_batch_size, batch_size, n_batched_models_train):
        self.num_meta_train_tasks = len(meta_train_data)
        self.meta_train_sampler = MetaDatasetSampler(meta_train_data, batch_size, meta_batch_size=meta_batch_size,
                                                     n_batched_models=n_batched_models_train, tiled=False)
        self.x_mean, self.y_mean, self.x_std, self.y_std = self.meta_train_sampler.get_standardization_stats()
        self.input_dim = self.meta_train_sampler.input_dim
        self.output_dim = self.meta_train_sampler.output_dim

    def _predict(self, x, sample_functions=True, sample_from_prior=True):
        if x.ndim == 2:
            x, _ = self.process_eval_batch(x, None)

        assert sample_functions and sample_from_prior
        nn_params, likelihood_std = self._sample_params_from_prior()

        y_pred = self.nn_model.call_parametrized(x, nn_params)
        pred_dist = self.likelihood.get_pred_mixture_dist(y_pred, std=likelihood_std)

        # un-standardize
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self.affine_transform.apply(pred_dist)

        return y_pred, pred_dist

    def _normalize_data(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y):
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist):
        return self.affine_transform.apply(pred_dist)

    @staticmethod
    def _handle_input_data(x, y=None, convert_to_tensor=True, dtype=tf.float32):
        if x.ndim == 1:
            x = tf.expand_dims(x, -1)

        assert x.ndim == 2

        if y is not None:
            if y.ndim == 1:
                y = tf.expand_dims(y, -1)
            assert x.shape[0] == y.shape[0]
            assert y.ndim == 2

            if convert_to_tensor:
                x, y = tf.cast(x, dtype=dtype), tf.cast(y, dtype=dtype)
            return x, y
        else:
            if convert_to_tensor:
                x = tf.cast(x, dtype=dtype)
            return x

    def _get_normalization_stats_dict(self):
        return dict(x_mean=self.x_mean, x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std)