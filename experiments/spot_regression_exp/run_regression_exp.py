import time
import json
import os
import argparse
import jax
import jax.numpy as jnp
import sys
import copy
import datetime
import wandb
from typing import List, Union
import types


from experiments.util import hash_dict, NumpyArrayEncoder
from experiments.data_provider import provide_data_and_sim, DATASET_CONFIGS
from sim_transfer.models import (
    BNN_SVGD,
    BNN_FSVGD,
    BNN_FSVGD_SimPrior,
    BNN_MMD_SimPrior,
    BNN_SVGD_DistillPrior,
    BNNGreyBox,
)
from sim_transfer.sims.simulators import (
    AdditiveSim,
    GaussianProcessSim,
    PredictStateChangeWrapper,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

ACTIVATION_DICT = {
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
    "elu": jax.nn.elu,
    "softplus": jax.nn.softplus,
    "swish": jax.nn.swish,
}

OUPUTSCALE_SPOT = [
    0.2,  # base_x
    0.2,  # base_y
    0.02,  # base_theta_sin
    0.02,  # base_theta_cos
    0.02,  # base_vel_x
    0.02,  # base_vel_y
    0.02,  # base_ang_vel
    0.2,  # ee_x
    0.2,  # ee_y
    0.02,  # ee_z
    0.02,  # ee_vx
    0.02,  # ee_vy
    0.02,  # ee_vz
]

# OUPUTSCALE_SPOT = None


def regression_experiment(
    # data parameters
    data_source: str,
    num_samples_train: int,
    data_seed: int = 981648,
    pred_diff: bool = False,
    # logging parameters
    use_wandb: bool = True,
    plot_traj_rollout: bool = True,
    # standard BNN parameters
    model: str = "BNN_SVGD",
    model_seed: int = 892616,
    likelihood_std: Union[List[float], float] = 0.1,
    data_batch_size: int = 8,
    min_train_steps: int = 2500,
    num_epochs: int = 60,
    max_train_steps: int = 100_000,
    num_sim_model_train_steps: int = 5_000,
    lr: float = 1e-3,
    hidden_activation: str = "leaky_relu",
    num_layers: int = 3,
    layer_size: int = 64,
    normalize_likelihood_std: bool = False,
    learn_likelihood_std: bool = False,
    likelihood_exponent: float = 1.0,
    likelihood_reg: float = 0.0,
    # SVGD parameters
    num_particles: int = 20,
    bandwidth_svgd: float = 10.0,
    weight_prior_std: float = 0.5,
    bias_prior_std: float = 1e1,
    # FSVGD parameters
    bandwidth_gp_prior: float = 0.4,
    num_measurement_points: int = 32,
    # FSVGD_Sim_Prior parameters
    bandwidth_score_estim: float = None,
    ssge_kernel_type: str = "IMQ",
    num_f_samples: int = 128,
    switch_score_estimator_frac: float = 0.75,
    added_gp_lengthscale: float = 5.0,
    added_gp_outputscale: Union[List[float], float] = 0.05,
    # BNN_SVGD_DistillPrior
    num_distill_steps: int = 500000,
):
    model_name = copy.deepcopy(model)

    num_train_steps = min(
        num_epochs * num_samples_train // data_batch_size + min_train_steps,
        max_train_steps,
    )

    # provide data and sim
    x_train, y_train, x_test, y_test, sim = provide_data_and_sim(
        data_source=data_source,
        data_spec={"num_samples_train": num_samples_train},
        data_seed=data_seed,
    )

    # handle pred diff mode
    if pred_diff:
        assert (
            x_train.shape[-1] == sim.input_size and y_train.shape[-1] == sim.output_size
        )
        y_train = y_train - x_train[..., : sim.output_size]
        y_test = y_test - x_test[..., : sim.output_size]
        sim = PredictStateChangeWrapper(sim)

    # toggle added GP
    if model.endswith("_no_add_gp"):
        no_added_gp = True
        model = model.replace("_no_add_gp", "")
        added_gp_outputscale = 0.0
    elif model in ["GreyBox", "SysID", "GreyBox_hf", "SysID_hf", "OnlySim_tuned"]:
        no_added_gp = True
    else:
        no_added_gp = False

    # create additive sim with a GP on top of the sim prior to model the fidelity gap
    if no_added_gp:
        sim = sim
    else:
        sim = AdditiveSim(
            base_sims=[
                sim,
                GaussianProcessSim(
                    sim.input_size,
                    sim.output_size,
                    output_scale=added_gp_outputscale,
                    length_scale=added_gp_lengthscale,
                ),
            ]
        )

    # setup standard model params
    standard_model_params = {
        "input_size": sim.input_size,
        "output_size": sim.output_size,
        "normalization_stats": sim.normalization_stats,
        "normalize_data": True,
        "rng_key": jax.random.PRNGKey(model_seed),
        "likelihood_std": likelihood_std,
        "data_batch_size": data_batch_size,
        "num_train_steps": num_train_steps,
        "lr": lr,
        "hidden_activation": ACTIVATION_DICT[hidden_activation],
        "hidden_layer_sizes": [layer_size] * num_layers,
        "normalize_likelihood_std": normalize_likelihood_std,
        "learn_likelihood_std": bool(learn_likelihood_std),
        "likelihood_exponent": likelihood_exponent,
    }

    if model == "BNN_SVGD":
        model = BNN_SVGD(
            num_particles=num_particles,
            bandwidth_svgd=bandwidth_svgd,
            weight_prior_std=weight_prior_std,
            bias_prior_std=bias_prior_std,
            likelihood_reg=likelihood_reg,
            **standard_model_params,
        )
    elif model == "BNN_FSVGD":
        model = BNN_FSVGD(
            domain=sim.domain,
            num_particles=num_particles,
            bandwidth_svgd=bandwidth_svgd,
            bandwidth_gp_prior=bandwidth_gp_prior,
            likelihood_reg=likelihood_reg,
            num_measurement_points=num_measurement_points,
            **standard_model_params,
        )
    elif "BNN_FSVGD_SimPrior" in model:
        score_estimator = model.split("_")[-1]
        assert score_estimator in [
            "SSGE",
            "ssge",
            "GP",
            "gp",
            "KDE",
            "kde",
            "nu-method",
            "gp+nu-method",
        ]
        model = BNN_FSVGD_SimPrior(
            domain=sim.domain,
            function_sim=sim,
            num_particles=num_particles,
            bandwidth_svgd=bandwidth_svgd,
            num_measurement_points=num_measurement_points,
            bandwidth_score_estim=bandwidth_score_estim,
            ssge_kernel_type=ssge_kernel_type,
            num_f_samples=num_f_samples,
            score_estimator=score_estimator,
            switch_score_estimator_frac=switch_score_estimator_frac,
            **standard_model_params,
        )
    elif model in ["GreyBox", "SysID", "GreyBox_hf", "SysID_hf"]:
        base_bnn = BNN_FSVGD(
            domain=sim.domain,
            num_particles=num_particles,
            bandwidth_svgd=bandwidth_svgd,
            bandwidth_gp_prior=bandwidth_gp_prior,
            likelihood_reg=likelihood_reg,
            num_measurement_points=num_measurement_points,
            **standard_model_params,
        )
        model = BNNGreyBox(
            base_bnn=base_bnn,
            sim=sim,
            use_base_bnn=(model == "GreyBox"),
            num_sim_model_train_steps=num_sim_model_train_steps,
        )
    elif model == "OnlySim_tuned":
        model = sim

        def fit_with_scan(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            log_to_wandb=False,
            log_period=1000,
        ):
            pass

        def eval(self, x_test, y_test, per_dim_metrics=False):
            return None

        def predict(self, x):
            y_pred = model._typical_f(x)
            return y_pred, None

        model.fit_with_scan = types.MethodType(fit_with_scan, model)
        model.eval = types.MethodType(eval, model)
        model.predict = types.MethodType(predict, model)

    else:
        raise NotImplementedError("Model {model} not implemented")

    # train model
    model.fit_with_scan(
        x_train, y_train, x_test, y_test, log_to_wandb=use_wandb, log_period=1000
    )
    # eval model
    eval_metrics = model.eval(x_test, y_test, per_dim_metrics=True)

    """----------------------------- EXTRA EVALUATION -----------------------------"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sim_transfer.sims.util import encode_angles as encode_angles_fn
    from sim_transfer.sims.util import decode_angles as decode_angles_fn
    from experiments.data_provider import _load_spot_datasets
    from sim_transfer.sims.util import delay_and_stack_spot_actions

    # load measured data for testing and eval
    dir_path = (
        "/home/bhoffman/Documents/MT_FS24/simulation_transfer/data/test_data_spot"
    )
    eval_trajectories_paths = sorted(
        [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".pickle")
        ]
    )
    eval_trajectories, eval_trajectories_id = _load_spot_datasets(eval_trajectories_paths), [
        os.path.basename(f).split(".")[0] for f in eval_trajectories_paths
    ]

    # extra evaluation settings
    action_delay_base = 2
    action_delay_ee = 1
    action_stacking = True if data_source == "spot_real_actionstack" else False
    step_range = min(200, min([traj[0].shape[0] for traj in eval_trajectories]))
    state_labels = [
        "base_x",
        "base_y",
        "base_theta",
        "base_vel_x",
        "base_vel_y",
        "base_ang_vel",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_vx",
        "ee_vy",
        "ee_vz",
    ]

    # setup extra eval metrics (rmse over all trajectories)
    extra_eval_metrics = {}

    # iterate over eval trajectories
    for traj, traj_id in zip(eval_trajectories, eval_trajectories_id):
        testing_x_pre_org, testing_u_pre_org, testing_y_org = traj

        # cut data
        testing_x_pre = testing_x_pre_org[:step_range]
        testing_u_pre = testing_u_pre_org[:step_range]
        testing_y = testing_y_org[:step_range]

        # apply action delay
        testing_u_pre = delay_and_stack_spot_actions(
            testing_u_pre,
            action_stacking,
            action_delay_base,
            action_delay_ee,
        )

        # prepare data
        testing_x_pre = encode_angles_fn(testing_x_pre, 2)
        testing_x = jnp.concatenate([testing_x_pre, testing_u_pre], axis=1)

        # prepare error metrics
        model_errors_max_ee = {}
        model_errors_total_ee = {}
        model_errors_max_base = {}
        model_errors_total_base = {}
        model_errors_max_base_theta = {}
        model_errors_total_base_theta = {}

        # simulate trajectory with learned model
        multistep = True
        if multistep and model_name != "DynamicsModel":
            y_pred_testing = []
            x_state = testing_x[0:1]
            for i in range(testing_x.shape[0]):
                y_pred, _ = model.predict(x_state)
                y_pred_testing.append(y_pred[0])
                if i < testing_x.shape[0] - 1:
                    u_next = testing_u_pre[i + 1 : i + 2]
                    x_state = jnp.concatenate([y_pred, u_next], axis=1)
            y_pred_testing = jnp.array(y_pred_testing)
        else:
            y_pred_testing, _ = model.predict(testing_x)
        y_pred_testing = decode_angles_fn(y_pred_testing, 2)

        # calculate errors for plotting
        ee_pos_error_running = jnp.linalg.norm(
            y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1
        )
        ee_pos_error_cumulative = jnp.cumsum(ee_pos_error_running)
        ee_pos_error_max = jnp.max(ee_pos_error_running)
        ee_pos_error_total = jnp.sum(ee_pos_error_running)
        model_errors_max_ee[model_name] = ee_pos_error_max
        model_errors_total_ee[model_name] = ee_pos_error_total

        base_pos_error_running = jnp.linalg.norm(
            y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1
        )
        base_pos_error_cumulative = jnp.cumsum(base_pos_error_running)
        base_pos_error_max = jnp.max(base_pos_error_running)
        base_pos_error_total = jnp.sum(base_pos_error_running)
        model_errors_max_base[model_name] = base_pos_error_max
        model_errors_total_base[model_name] = base_pos_error_total

        base_theta_error_running = jnp.linalg.norm(
            y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1
        )
        base_theta_error_cumulative = jnp.cumsum(base_theta_error_running)
        base_theta_error_max = jnp.max(base_theta_error_running)
        base_theta_error_total = jnp.sum(base_theta_error_running)
        model_errors_max_base_theta[model_name] = base_theta_error_max
        model_errors_total_base_theta[model_name] = base_theta_error_total

        # fill extra evals metrics for current trajectory
        for state_label in state_labels:
            state_idx = state_labels.index(state_label)
            state_error = (y_pred_testing[:, state_idx] - testing_y[:, state_idx]) ** 2
            extra_eval_metrics[f"{state_label}_rmse"] = (
                extra_eval_metrics.get(
                    f"{state_label}_rmse", jnp.zeros(state_error.shape)
                )
                + state_error
            )
        base_pos_error = jnp.linalg.norm(y_pred_testing[:, 0:2] - testing_y[:, 0:2], axis=1) ** 2
        theta_error = jnp.linalg.norm(y_pred_testing[:, 2:3] - testing_y[:, 2:3], axis=1) ** 2
        ee_pos_error = jnp.linalg.norm(y_pred_testing[:, 6:9] - testing_y[:, 6:9], axis=1) ** 2
        extra_eval_metrics["base_pos_rmse"] = extra_eval_metrics.get("base_pos_rmse", jnp.zeros(base_pos_error.shape)) + base_pos_error
        extra_eval_metrics["theta_rmse"] = extra_eval_metrics.get("theta_rmse", jnp.zeros(base_theta_error_running.shape)) + theta_error
        extra_eval_metrics["ee_pos_rmse"] = extra_eval_metrics.get("ee_pos_rmse", jnp.zeros(ee_pos_error.shape)) + ee_pos_error

        # detailed plot of trajectory rollout and errors
        if plot_traj_rollout:
            # prepare plots
            fig, axs = plt.subplots(6, 2, figsize=(30, 15))
            fig_ee_error, axs_ee_error = plt.subplots(4, 1, figsize=(30, 15))
            fig_base_error, axs_base_error = plt.subplots(4, 2, figsize=(30, 15))

            # plot true data
            for i in range(6):
                axs[i, 0].plot(testing_y[:, i], label="true")
                axs[i, 0].set_title(state_labels[i])

                if i + 6 < testing_y.shape[-1]:
                    axs[i, 1].plot(testing_y[:, i + 6], label="true")
                    axs[i, 1].set_title(state_labels[i + 6])

            # prepare error plots
            axs_ee_error[0].set_title("Running ee position error")
            axs_ee_error[1].set_title("Running cumulative ee position error")
            axs_ee_error[2].set_title("Max ee position error")
            axs_ee_error[3].set_title("Total cumulative ee position error")
            axs_base_error[0, 0].set_title("Running base position error")
            axs_base_error[1, 0].set_title("Running cumulative base position error")
            axs_base_error[2, 0].set_title("Max base position error")
            axs_base_error[3, 0].set_title("Total cumulative base position error")
            axs_base_error[0, 1].set_title("Running base theta error")
            axs_base_error[1, 1].set_title("Running cumulative base theta error")
            axs_base_error[2, 1].set_title("Max base theta error")
            axs_base_error[3, 1].set_title("Total cumulative base theta error")

            # plot simulated trajectory
            for i in range(6):
                axs[i, 0].plot(
                    y_pred_testing[:, i], label=f"pred {model_name}", linestyle="--"
                )

                if i + 6 < y_test.shape[-1]:
                    axs[i, 1].plot(
                        y_pred_testing[:, i + 6],
                        label=f"pred {model_name}",
                        linestyle="--",
                    )

            # plot running and cumulative errors
            axs_ee_error[0].plot(ee_pos_error_running, label=f"{model_name}")
            axs_ee_error[1].plot(ee_pos_error_cumulative, label=f"{model_name}")

            axs_base_error[0, 0].plot(base_pos_error_running, label=f"{model_name}")
            axs_base_error[1, 0].plot(base_pos_error_cumulative, label=f"{model_name}")

            axs_base_error[0, 1].plot(base_theta_error_running, label=f"{model_name}")
            axs_base_error[1, 1].plot(
                base_theta_error_cumulative, label=f"{model_name}"
            )

            # plot max and total errors
            axs_ee_error[2].barh(
                list(model_errors_max_ee.keys()), list(model_errors_max_ee.values())
            )
            axs_ee_error[3].barh(
                list(model_errors_total_ee.keys()), list(model_errors_total_ee.values())
            )
            axs_base_error[2, 0].barh(
                list(model_errors_max_base.keys()), list(model_errors_max_base.values())
            )
            axs_base_error[3, 0].barh(
                list(model_errors_total_base.keys()),
                list(model_errors_total_base.values()),
            )
            axs_base_error[2, 1].barh(
                list(model_errors_max_base_theta.keys()),
                list(model_errors_max_base_theta.values()),
            )
            axs_base_error[3, 1].barh(
                list(model_errors_total_base_theta.keys()),
                list(model_errors_total_base_theta.values()),
            )

            for i in range(6):
                axs[i, 0].legend(fontsize=8)
                axs[i, 1].legend(fontsize=8)

            for i in range(4):
                axs_ee_error[i].legend(fontsize=8)
                axs_base_error[i, 0].legend(fontsize=8)
                axs_base_error[i, 1].legend(fontsize=8)

            if use_wandb:
                wandb.log(
                    {
                        f"extra_eval_plots/{traj_id}/plot": wandb.Image(fig),
                        f"extra_eval_plots/{traj_id}/ee_error_plot": wandb.Image(fig_ee_error),
                        f"extra_eval_plots/{traj_id}/base_error_plot": wandb.Image(
                            fig_base_error
                        ),
                    }
                )
 

    # calculate rmse across all eval trajectories
    for state_label in state_labels:
        state_idx = state_labels.index(state_label)
        extra_eval_metrics[f"{state_label}_rmse"] = jnp.sqrt(
            extra_eval_metrics[f"{state_label}_rmse"] / len(eval_trajectories)
        )
    extra_eval_metrics["base_pos_rmse"] = jnp.sqrt(extra_eval_metrics["base_pos_rmse"] / len(eval_trajectories))
    extra_eval_metrics["theta_rmse"] = jnp.sqrt(extra_eval_metrics["theta_rmse"] / len(eval_trajectories))
    extra_eval_metrics["ee_pos_rmse"] = jnp.sqrt(extra_eval_metrics["ee_pos_rmse"] / len(eval_trajectories))

    # log extra eval metrics
    if use_wandb:
        for step in range(step_range):
            step_extra_eval_metrics = {
                f"extra_eval/{state_label}_rmse": float(
                    extra_eval_metrics[f"{state_label}_rmse"][step]
                )
                for state_label in state_labels
            }
            step_extra_eval_metrics.update(
                {
                    "extra_eval/base_pos_rmse": float(extra_eval_metrics["base_pos_rmse"][step]),
                    "extra_eval/theta_rmse": float(extra_eval_metrics["theta_rmse"][step]),
                    "extra_eval/ee_pos_rmse": float(extra_eval_metrics["ee_pos_rmse"][step]),
                    "extra_eval/step": step,
                }
            )
            wandb.log(step_extra_eval_metrics)

    return eval_metrics


def main(args):
    """"""

    """ generate experiment hash and set up redirect of output streams """
    exp_params = args.__dict__
    exp_result_folder = exp_params.pop("exp_result_folder")
    use_wandb = exp_params.pop("use_wandb")
    exp_name = exp_params.pop("exp_name")
    exp_hash = hash_dict(exp_params)

    if exp_result_folder is not None:
        os.makedirs(exp_result_folder, exist_ok=True)

    if "added_gp_outputscale" in exp_params:
        factor = 1
        if exp_params["added_gp_outputscale"] > 0:
            factor = exp_params["added_gp_outputscale"]
        if "spot" in exp_params["data_source"] and OUPUTSCALE_SPOT is not None:
            outputscales_spot = factor * jnp.array(OUPUTSCALE_SPOT)
            exp_params["added_gp_outputscale"] = outputscales_spot.tolist()
        else:
            if exp_params["added_gp_outputscale"] < 0:
                raise AssertionError("passed negative value for added_gp_outputscale")

    # set likelihood_std to default value if not specified
    if exp_params["likelihood_std"] is None:
        likelihood_std = DATASET_CONFIGS[args.data_source]["likelihood_std"]["value"]
        exp_params["likelihood_std"] = likelihood_std
        print(
            f"Setting likelihood_std to data_source default value from DATASET_CONFIGS "
            f"which is {exp_params['likelihood_std']}"
        )

    from pprint import pprint

    print("\nExperiment parameters:")
    pprint(exp_params)
    print("")

    """ Experiment core """
    t_start = time.time()

    if use_wandb:
        # hash of experiments without seeds
        exp_params_no_seeds = copy.deepcopy(exp_params)
        [exp_params_no_seeds.pop(k) for k in ["model_seed", "data_seed"]]
        exp_hash_no_seeds = hash_dict(exp_params_no_seeds)

        wandb.init(
            project="spot_model_regression_sweep_v1",
            config=exp_params,
            name=f"{exp_name}/{args.data_source}/{args.model}/{exp_hash}",
            group=f"{exp_name}/{args.data_source}/{args.model}/{exp_hash_no_seeds}",
        )

    eval_metrics = regression_experiment(**exp_params, use_wandb=use_wandb)

    t_end = time.time()

    if use_wandb:
        for key, val in eval_metrics.items():
            wandb.summary[key] = float(val)
        wandb.log({f"final_{key}": float(val) for key, val in eval_metrics.items()})

    """ Save experiment results and configuration """
    results_dict = {
        "evals": eval_metrics,
        "params": exp_params,
        "duration_total": t_end - t_start,
    }

    if exp_result_folder is None:
        from pprint import pprint

        pprint(results_dict)
    else:
        exp_result_file = os.path.join(exp_result_folder, f"{exp_hash}.json")
        with open(exp_result_file, "w") as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print(f"Dumped results to {exp_result_file}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    current_date = datetime.datetime.now().strftime("%b%d").lower()
    parser = argparse.ArgumentParser(description="Meta-BO run")

    # prevent TF from reserving all GPU memory
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # general args
    parser.add_argument("--exp_result_folder", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=f"test_{current_date}")
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--plot_traj_rollout", type=bool, default=True)

    # data parameters
    parser.add_argument("--data_source", type=str, default="spot_real_actionstack")
    parser.add_argument("--pred_diff", type=int, default=0)
    parser.add_argument("--num_samples_train", type=int, default=4_400)
    parser.add_argument("--data_seed", type=int, default=107030)

    # standard BNN parameters
    parser.add_argument("--model", type=str, default="OnlySim_tuned")
    parser.add_argument("--model_seed", type=int, default=922852)
    parser.add_argument("--likelihood_std", type=float, default=None)
    parser.add_argument("--learn_likelihood_std", type=int, default=1)
    parser.add_argument("--likelihood_reg", type=float, default=0.0)
    parser.add_argument("--data_batch_size", type=int, default=32)
    parser.add_argument("--min_train_steps", type=int, default=10_000)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--max_train_steps", type=int, default=100_000)
    parser.add_argument("--num_sim_model_train_steps", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_activation", type=str, default="leaky_relu")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--layer_size", type=int, default=64)
    parser.add_argument("--normalize_likelihood_std", type=bool, default=True)
    parser.add_argument("--likelihood_exponent", type=float, default=1.0)

    # SVGD parameters
    parser.add_argument("--num_particles", type=int, default=20)
    parser.add_argument("--bandwidth_svgd", type=float, default=5.0) # vary
    parser.add_argument("--weight_prior_std", type=float, default=0.5)
    parser.add_argument("--bias_prior_std", type=float, default=1.0)

    # FSVGD parameters
    parser.add_argument("--bandwidth_gp_prior", type=float, default=1.0) # vary: increase to larger for less overfitting
    parser.add_argument("--num_measurement_points", type=int, default=32) # similar to batchsize

    # FSVGD_SimPrior parameters
    parser.add_argument("--bandwidth_score_estim", type=float, default=None)
    parser.add_argument("--ssge_kernel_type", type=str, default="IMQ")
    parser.add_argument("--num_f_samples", type=int, default=2048)
    parser.add_argument("--switch_score_estimator_frac", type=float, default=0.75)

    # Additive SimPrior GP parameters
    parser.add_argument("--added_gp_lengthscale", type=float, default=5.0)
    parser.add_argument("--added_gp_outputscale", type=float, default=1.0)

    # FSVGD_SimPrior parameters
    parser.add_argument("--num_distill_steps", type=int, default=50000)

    args = parser.parse_args()
    main(args)


