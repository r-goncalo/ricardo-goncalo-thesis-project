

import os
from automl.consts import CONFIGURATION_FILE_NAME
from automl.loggers.result_logger import ResultLogger, RESULTS_FILENAME
from automl.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline, OPTUNA_STUDY_PATH, BASE_CONFIGURATION_NAME
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.utils.json_utils.json_component_utils import dict_from_path, gen_component_from
from automl.utils.optuna_utils import load_study_from_database
from optuna.importance import get_param_importances
import optuna
from automl.loggers.global_logger import globalWriteLine
from automl.utils.files_utils import get_first_path_with_name
import pandas as pd



def get_hp_opt_results_logger(experiment_path, results_filename=RESULTS_FILENAME) -> ResultLogger:

    hp_results_columns = HyperparameterOptimizationPipeline.results_columns

    hyperparameter_optimization_results : ResultLogger = ResultLogger(input={
                                        "base_directory" : experiment_path,
                                        "artifact_relative_directory" : '',
                                        "results_filename" : RESULTS_FILENAME,
                                        "results_columns" : hp_results_columns,
                                        "create_new_directory" : False
                                      })

    hyperparameter_optimization_results.proccess_input_if_not_processed()

    return hyperparameter_optimization_results




def get_hp_opt_optuna_study(hp_results_logger : ResultLogger, database_path=OPTUNA_STUDY_PATH):
    optuna_study = load_study_from_database(database_path=hp_results_logger.get_artifact_directory() + '\\' + database_path)

    return optuna_study


def get_evaluation_statistics_per_step(hp_results_logger : ResultLogger):
    '''
    Returns, for each step, the average mean results of trials and average standard deviation of results of trials
    That is, the standard deviation and mean of results for the same experiment (different component indexes) and step
    '''

    df = hp_results_logger.get_dataframe()

    if df.empty:
        return {}

    # Step 1: stats per (experiment, step)
    exp_step_stats = (
        df.groupby(["experiment", "step"])["result"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Step 2: aggregate per step
    step_stats = (
        exp_step_stats.groupby("step")
        .agg(
            avg_mean=("mean", "mean"),
            avg_std=("std", "mean")
        )
        .reset_index()
    )

    # Convert to dictionary {step: {avg_mean, avg_std}}
    statistics_per_step = {
        int(row["step"]): {
            "avg_mean": float(row["avg_mean"]),
            "avg_std": float(row["avg_std"]) if not pd.isna(row["avg_std"]) else 0.0
        }
        for _, row in step_stats.iterrows()
    }

    return statistics_per_step



def get_component_stats_by_step(component_path):

    configuration_dict = dict_from_path(os.path.join(component_path, CONFIGURATION_FILE_NAME))

    raise NotImplementedError()


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


#def get_params_in_optuna(optuna_study):
#
#    # DataFrame from study
#    df = optuna_study.trials_dataframe()
#    # Hyperparameter columns
#    param_cols = [c for c in df.columns if c.startswith("params_")]
#
#    return param_cols
#
#
#def plot_scattered_values_for_param(optuna_study, highlight_trials, param):
#
#    if highlight_trials is None:
#        highlight_trials = []
#
#
#def plot_scattered_values_for_params(optuna_study, highlight_trials, params=None):
#
#    if params is None:
#        return plot_scattered_values_for_all_params(optuna_study, highlight_trials)
#    
#    elif isinstance(params, list):
#        for param in params:
#            plot_scattered_values_for_param(optuna_study, highlight_trials, param)
#
#    else:
#        plot_scattered_values_for_param(optuna_study, highlight_trials, params)

def plot_scattered_values_for_all_params(optuna_study, highlight_trials=None):

    import optuna
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    if highlight_trials is None:
        highlight_trials = []

    # DataFrame from study
    df = optuna_study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    # Hyperparameter columns
    param_cols = [c for c in df.columns if c.startswith("params_")]

    # Prepare highlighted trial data
    highlighted_points = []
    for tnum in highlight_trials:
        try:
            t = optuna_study.trials[tnum]
            highlighted_points.append(t)
        except:
            print(f"Warning: trial {tnum} not found, skipping.")

    trial_numbers = df["number"].values

    # Normalize trial numbers to [0,1]
    min_trial = trial_numbers.min()
    max_trial = trial_numbers.max()
    
    norm_trials =  ( (trial_numbers - min_trial) / (max_trial - min_trial + 1e-8) )
    highlight_norm_trials = ( (trial_numbers - min_trial) / (max_trial - min_trial + 1e-8) ) 

    # Invert so earlier trials are more transparent
    alphas = 0.1 + 0.9 * norm_trials  # alpha between 0.1 and 1.0


    # Plot each parameter
    for param in param_cols:
        plt.figure(figsize=(7, 5))

        x = df[param].values
        y = df["value"].values

        zipped_all_trials = iter(zip(x, y, alphas))

        xi, yi, ai = next(zipped_all_trials)
        plt.scatter(xi, yi, alpha=ai, color="red", label="All trials")

        # Base plot: all trials
        for xi, yi, ai in zipped_all_trials:
            plt.scatter(xi, yi, alpha=ai, color="red")
        

        base_name = param.replace("params_", "")

        # Collect all highlighted X,Y
        hl_x = []
        hl_y = []
        for t in highlighted_points:
            if base_name in t.params:
                hl_x.append(t.params[base_name])
                hl_y.append(t.value)

        # Scatter highlighted points (if exist)
        if len(hl_x) > 0:
            plt.scatter(
                hl_x, hl_y,
                color="blue",
                s=70,
                label="Highlighted trials"
            )

        plt.xlabel(base_name)
        plt.ylabel("Objective")
        plt.title(f"{base_name} vs Objective (highlighted trials as a group)")
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
                           aggregate_number=10,
                                    study_evaluations=True,
                            ):


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

    if study_evaluations:

        evaluations_logger = get_evaluations_results_logger(os.path.join(results_logger.get_artifact_directory(), ".."))

        if evaluations_logger != None:
            study_of_evaluations(configuration_name, evaluations_logger)

        else:
            print(f"Study has no evaluations to study")


def study_of_components_for_configuration(configuration_name : str, results_loggers : dict[str, ResultLogger],
                           #x_axis_to_use='episode',
                           x_axis_to_use='total_steps',
                           y_axis_to_use='episode_reward',
                           aggregate_number=10,
                            colors_for_component_indexes : dict = None
                            ):

    
    for component_name, results_logger in results_loggers.items():

            color = colors_for_component_indexes[int(component_name)] if colors_for_component_indexes is not None else None

            results_logger.plot_confidence_interval(x_axis=x_axis_to_use, 
                                            y_column=y_axis_to_use,
                                            show_std=True, 
                                            to_show=False, 
                                            y_values_label=f"mov_avg_std_{component_name} ({aggregate_number})", 
                                            aggregate_number=aggregate_number,
                                            color=color,
                                            alpha=0.1)   



    results_logger.plot_current_graph(title=configuration_name, y_label=y_axis_to_use)     



def get_evaluations_path(base_path):

    return get_first_path_with_name(base_path, "evaluations")


def get_evaluations_results_logger(base_path):

    evaluations_path = get_evaluations_path(base_path)

    if evaluations_path == None:
        return None
    

    return ResultLogger(input={
                                                "base_directory" : evaluations_path,
                                                "artifact_relative_directory" : '',
                                                "create_new_directory" : False,
                                                "results_filename" : "evaluations.csv"
                                              })




def study_of_evaluations(configuration_name : str, results_logger : ResultLogger,
                           #x_axis_to_use='episode',
                           x_axis_to_use='evaluation',
                           y_axis_to_use='result'
                           ):
    
    results_logger.plot_bar_graph(x_axis=x_axis_to_use, y_axis=y_axis_to_use, to_show=False)

    results_logger.plot_linear_regression(x_axis='evaluation', y_axis='result', to_show=False, color="orange")


    results_logger.plot_current_graph(title=f"{configuration_name}_evaluations", y_label=y_axis_to_use, y_min=0)


def get_results_of_configuration_in_path(configuration_path, 
                                         configurations_results_relative_path,
                                         results_path=RESULTS_FILENAME) -> ResultLogger:

        results_logger_of_config = ResultLogger(input={
                                    "results_filename" : results_path,
                                    "base_directory" : os.path.join(configuration_path, configurations_results_relative_path),
                                    "artifact_relative_directory" : '',
                                    "create_new_directory" : False

                                  })

        results_logger_of_config.proccess_input()

        return results_logger_of_config



def get_results_of_configurations(experiment_path,
                                   base_configuration_name=BASE_CONFIGURATION_NAME,
                                  results_path=RESULTS_FILENAME) -> dict[str, ResultLogger]:
    
    '''Gets directly the results of configurations in a path'''

    results_of_configurations : dict[str, ResultLogger] = {}

    configurations_results_relative_path = "RLTrainerComponent"

    for configuration_name in os.listdir(experiment_path):

        if configuration_name.startswith(base_configuration_name):

            configuration_path = os.path.join(experiment_path, configuration_name)

            if os.path.isdir(configuration_path):  # Ensure it's a file, not a subdirectory

                try:
                    results_of_configurations[configuration_name] = get_results_of_configuration_in_path(
                        configuration_path,
                        configurations_results_relative_path,
                        results_path
                    )

                except Exception as e:
                    print(f"Did not manage to get configuration {configuration_name} due to error {e}")

            else:
                print(f"WARNING: Configuration path with name {configuration_name} is not a directory")

            
    return results_of_configurations


def get_results_of_configurations_components(experiment_path,
                                   base_configuration_name=BASE_CONFIGURATION_NAME,
                                   configurations_results_relative_path = "RLTrainerComponent",
                                  results_path=RESULTS_FILENAME,
                                  use_multiple=True) -> dict[str, dict[str, ResultLogger]]:

    results_of_configurations : dict[str, dict[str, ResultLogger]] = {}

    for configuration_name in os.listdir(experiment_path):

        if configuration_name.startswith(base_configuration_name):

            configuration_path = os.path.join(experiment_path, configuration_name)

            if os.path.isdir(configuration_path):  # Ensure it's a file, not a subdirectory

                configuration_dict = {}

                if use_multiple:

                    for configuration_component_name in os.listdir(configuration_path):

                        configuration_component_path = os.path.join(configuration_path, configuration_component_name)

                        try:
                            configuration_dict[configuration_component_name] = get_results_of_configuration_in_path(
                                configuration_component_path,
                                configurations_results_relative_path,
                                results_path
                            )

                        except Exception as e:
                            print(f"Did not manage to get configuration {configuration_name} due to error {e}")
                
                else:
                    configuration_dict["0"] = get_results_of_configuration_in_path(
                                configuration_path,
                                configurations_results_relative_path,
                                results_path
                            )

                results_of_configurations[configuration_name] = configuration_dict

            else:
                globalWriteLine(f"WARNING: Configuration path with name {configuration_name} is not a directory")

            
    return results_of_configurations


def get_pruned_trials(optuna_study):

    '''Gets pruned trials per steps, organized by steps'''

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

    return pruned_optuna_trials, pruned_optuna_trials_per_steps, pruned_trials


def get_trials_with_decreasing_intermediates(study):

    bad_trials = []

    for trial in study.trials:
        ivals = trial.intermediate_values

        # Need at least 2 intermediate values to compare
        if len(ivals) < 2:
            continue

        # Sort by step index (intermediate_values is a dict: step -> value)
        steps = sorted(ivals.keys())
        values = [ivals[s] for s in steps]

        # Check if any later value is lower than any earlier value
        ever_decreased = any(values[j] < values[i] 
                             for i in range(len(values)) 
                             for j in range(i + 1, len(values)))

        if ever_decreased:
            bad_trials.append(trial)

    return bad_trials

def print_intermidiate_values(trial_list : optuna.trial.FrozenTrial):

    import optuna

    if len(trial_list) == 0:
        return
    
    for trial in trial_list:

        # Sort the intermediate values by step index
        steps = sorted(trial.intermediate_values.keys())
        values = [trial.intermediate_values[s] for s in steps]

        # Format as: trial X: v1, v2, v3
        values_str = ", ".join(str(v) for v in values)
        print(f"trial {trial.number}: {values_str}")