

import os
from automl.loggers.result_logger import ResultLogger, RESULTS_FILENAME
from automl.meta_rl.hp_optimization_pipeline import HyperparameterOptimizationPipeline, OPTUNA_STUDY_PATH, BASE_CONFIGURATION_NAME
from automl.utils.optuna_utils import load_study_from_database
from optuna.importance import get_param_importances
import optuna



def get_hp_opt_results_logger(experiment_path, results_filename=RESULTS_FILENAME) -> ResultLogger:

    hp_results_columns = HyperparameterOptimizationPipeline.results_columns

    hyperparameter_optimization_results : ResultLogger = ResultLogger(input={
                                        "base_directory" : experiment_path,
                                        "artifact_relative_directory" : '',
                                        "results_filename" : RESULTS_FILENAME,
                                        "results_columns" : hp_results_columns,
                                        "create_new_directory" : False
                                      })

    hyperparameter_optimization_results.proccess_input_if_not_proccesd()

    return hyperparameter_optimization_results




def get_hp_opt_optuna_study(hp_results_logger : ResultLogger, database_path=OPTUNA_STUDY_PATH):
    optuna_study = load_study_from_database(database_path=hp_results_logger.get_artifact_directory() + '\\' + database_path)

    return optuna_study




def print_optuna_trials_info(optuna_study):

    print("\n===== Trials Info =====")
    for t in optuna_study.trials:
        print(f"Trial {t.number}:")
        print(f"  State: {t.state}")
        print(f"  Value: {t.value}")
        print(f"  Params: {t.params}")
        print(f"  User attrs: {t.user_attrs}")
        print(f"  System attrs: {t.system_attrs}")
        print("-" * 40)



def print_optuna_param_importances(optuna_study):

    importances = get_param_importances(optuna_study)

    # Print nicely
    for param, importance in importances.items():
        print(f"{param}: {importance:.4f}")



def plot_scattered_values_for_param(optuna_study):

    import optuna
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    # Assume `optuna_study` is already loaded
    df = optuna_study.trials_dataframe()  # columns: value, params_*, state, etc.

    # Only consider completed trials
    df = df[df['state'] == 'COMPLETE']

    # Get all hyperparameter columns
    param_cols = [c for c in df.columns if c.startswith("params_")]

    # Plot each hyperparameter vs objective with Gaussian Process regression
    for param in param_cols:
        plt.figure(figsize=(7, 5))

        # Extract values and objective
        x = df[param].values
        y = df['value'].values

        # Scatter plot: parameter vs objective
        plt.scatter(x, y, alpha=0.6, label="Trials")

        # Only fit GP if the param is numeric
        if np.issubdtype(x.dtype, np.number):
            # Reshape for sklearn (expects 2D arrays)
            X = x.reshape(-1, 1)
            Y = y.reshape(-1, 1)

            # Kernel: RBF (smooth function) + WhiteKernel (noise)
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
            gp.fit(X, Y)

            # Predict on smooth range
            x_range = np.linspace(min(x), max(x), 200).reshape(-1, 1)
            y_mean, y_std = gp.predict(x_range, return_std=True)

            # Plot GP mean
            plt.plot(x_range, y_mean, "r-", lw=2, label="GP mean")

            # Plot uncertainty band (±1 std)
            plt.fill_between(
                x_range.ravel(),
                (y_mean - y_std).ravel(),
                (y_mean + y_std).ravel(),
                color="r",
                alpha=0.2,
                label="GP ±1σ"
            )

        plt.xlabel(param)
        plt.ylabel("Objective value")
        plt.title(f"Effect of {param} on Objective (GP fit)")
        plt.legend()
        plt.grid(True)
        plt.show()


#def partial dependency:

#   import pandas as pd
#   
#   # Convert trials into a DataFrame
#   df = optuna_study.trials_dataframe(attrs=("number", "value", "params", "state"))
#   
#   # Keep only completed trials
#   df = df[df["state"] == "COMPLETE"]
#   
#   X = df[[c for c in df.columns if c.startswith("params_")]]
#   y = df["value"]
#   
#   from sklearn.ensemble import RandomForestRegressor
#   
#   # You could also use GradientBoosting, XGBoost, LightGBM, etc.
#   surrogate = RandomForestRegressor(n_estimators=200, random_state=0)
#   surrogate.fit(X, y)
#   
#   from sklearn.inspection import PartialDependenceDisplay
#   
#   # Single parameter PDP
#   PartialDependenceDisplay.from_estimator(surrogate, X, features=["params_learning_rate"])
#   
#   # 2D PDP for interaction between two hyperparams
#   PartialDependenceDisplay.from_estimator(
#       surrogate, X, features=[("params_learning_rate", "params_optimization_interval")]
#   )


def study_of_configuration(configuration_name : str, results_logger : ResultLogger,
                           #x_axis_to_use='episode',
                           x_axis_to_use='total_steps',
                           y_axis_to_use='episode_reward',
                           aggregate_number=10):


    #results_logger.plot_graph(x_axis='episode', y_axis=[('total_reward', name)], to_show=False)

    results_logger.plot_confidence_interval(x_axis=x_axis_to_use, 
                                            y_column=y_axis_to_use,
                                            show_std=True, 
                                            to_show=False, 
                                            y_values_label=f"mov_avg_std ({aggregate_number})", 
                                            aggregate_number=aggregate_number)

    #results_logger.plot_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False)

    #results_logger.plot_piecewise_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False, n_segments=10)

    results_logger.plot_polynomial_regression(x_axis=x_axis_to_use, y_axis=y_axis_to_use, to_show=False, degrees=4)


    results_logger.plot_current_graph(title=configuration_name, y_label=y_axis_to_use)


def get_results_of_configurations(experiment_path,
                                   base_configuration_name=BASE_CONFIGURATION_NAME,
                                  results_path=RESULTS_FILENAME) -> dict[str, ResultLogger]:

    results_of_configurations : dict[str, ResultLogger] = {}

    configurations_results_relative_path = "RLTrainerComponent"

    for configuration_name in os.listdir(experiment_path):

        if configuration_name.startswith(base_configuration_name):

            configuration_path = os.path.join(experiment_path, configuration_name)

            if os.path.isdir(configuration_path):  # Ensure it's a file, not a subdirectory

                try:
                    results_logger_of_config = ResultLogger(input={
                                                "results_filename" : results_path,
                                                "base_directory" : f"{configuration_path}\\{configurations_results_relative_path}",
                                                "artifact_relative_directory" : '',
                                                "create_new_directory" : False

                                              })

                    results_logger_of_config.proccess_input()

                    results_of_configurations[configuration_name] = results_logger_of_config

                except Exception as e:
                    print(f"Did not manage to store configuration {configuration_name} due to error {e}")

            else:
                print(f"WARNING: Configuration path with name {configuration_name} is not a directory")

            
    return results_of_configurations


def get_pruned_trials(optuna_study):

    pruned_optuna_trials = [trial for trial in optuna_study.trials if trial.state == optuna.trial.TrialState.PRUNED]

    pruned_optuna_trials_per_steps : dict[int, list[optuna.trial.FrozenTrial]] = {} #the pruned trials by the number of completed steps

    for pruned_optuna_trial in pruned_optuna_trials:

        n_completed_steps = len(pruned_optuna_trial.intermediate_values)

        try:
            list_of_pruned = pruned_optuna_trials_per_steps[n_completed_steps]

        except:
            list_of_pruned = []
            pruned_optuna_trials_per_steps[n_completed_steps] = list_of_pruned    

        list_of_pruned.append(pruned_optuna_trial)


    for list_of_pruned in pruned_optuna_trials_per_steps.values():
        list_of_pruned.sort(key=lambda trial: trial.value) 



    pruned_trials = [f'{BASE_CONFIGURATION_NAME}_{trial.number + 1}' for trial in optuna_study.trials if trial.state == optuna.trial.TrialState.PRUNED]

    print(f"Pruned trials: {pruned_trials}")

    return pruned_optuna_trials, pruned_optuna_trials_per_steps, pruned_trials