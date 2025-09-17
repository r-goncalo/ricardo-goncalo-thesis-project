


import os

from automl.utils.files_utils import open_or_create_folder
from automl.utils.json_component_utils import dict_from_json_string, json_string_of_component_dict


SCRIPT_PATH = "C:\\rgoncalo\\ricardo-goncalo-thesis-project\\other_code\\RunSimplyHpExp.bat"

def sb3_montaincar_semi_trained_1():


    # === Define the list of commands to run ===
    commands = [

        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_10 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_10 --TOOPTIMIZECONFIG to_optimize_configuration_10.json --CONFIG configuration_reduced.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_30 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_30 --TOOPTIMIZECONFIG to_optimize_configuration_30.json --CONFIG configuration_reduced.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_50 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_50 --TOOPTIMIZECONFIG to_optimize_configuration_50.json --CONFIG configuration_reduced.json',

        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_10 --EXPSTOREPATH sb3_montaincar_semi_trained_10 --TOOPTIMIZECONFIG to_optimize_configuration_10.json --CONFIG configuration.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_30 --EXPSTOREPATH sb3_montaincar_semi_trained_30 --TOOPTIMIZECONFIG to_optimize_configuration_30.json --CONFIG configuration.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_50 --EXPSTOREPATH sb3_montaincar_semi_trained_50 --TOOPTIMIZECONFIG to_optimize_configuration_50.json --CONFIG configuration.json',

        [SCRIPT_PATH, "--LOGBASENAME", "sb3_montaincar_semi_trained_reduced_75",
         "--EXPSTOREPATH", "sb3_montaincar_semi_trained_reduced_75",
         "--TOOPTIMIZECONFIG", "to_optimize_configuration_75.json",
         "--CONFIG", "configuration_reduced.json"],
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_100 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_100 --TOOPTIMIZECONFIG to_optimize_configuration_100.json --CONFIG configuration_reduced.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_150 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_150 --TOOPTIMIZECONFIG to_optimize_configuration_150.json --CONFIG configuration_reduced.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_reduced_200 --EXPSTOREPATH sb3_montaincar_semi_trained_reduced_200 --TOOPTIMIZECONFIG to_optimize_configuration_200.json --CONFIG configuration_reduced.json',

        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_75  --EXPSTOREPATH sb3_montaincar_semi_trained_75 --TOOPTIMIZECONFIG to_optimize_configuration_75.json --CONFIG configuration.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_100 --EXPSTOREPATH sb3_montaincar_semi_trained_100 --TOOPTIMIZECONFIG to_optimize_configuration_100.json --CONFIG configuration.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_150 --EXPSTOREPATH sb3_montaincar_semi_trained_150 --TOOPTIMIZECONFIG to_optimize_configuration_150.json --CONFIG configuration.json',
        f'{SCRIPT_PATH} --LOGBASENAME sb3_montaincar_semi_trained_200 --EXPSTOREPATH sb3_montaincar_semi_trained_200 --TOOPTIMIZECONFIG to_optimize_configuration_200.json --CONFIG configuration.json',
    ]

    return commands


def sb3_montaincar_semi_trained_2(directory_of_models, 
                                 directory_to_store_experiment,
                                 base_to_opt_config_path, 
                                 hp_opt_config_path):
    
    print(f"Running experiment sb3_montaincar_semi_trained_v2\n")

    print(f"Base configuration to optimize path: {base_to_opt_config_path}")

    print(f"Base configuration of Hyperparameter Optimization pipeline: {hp_opt_config_path}")

    print(f"\nDirectory of models: {directory_of_models}")


    # get model paths
    model_paths = [
        os.path.join(directory_of_models, f)
        for f in os.listdir(directory_of_models)
        if os.path.isfile(os.path.join(directory_of_models, f))
    ]

    model_names = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]

    print(f"Models found: {model_names}")

    directory_to_store_experiment = open_or_create_folder(directory_to_store_experiment, folder_name="sb3_montaincar_semi_trained_v2")
    print(f"\nDirectory to store experiment: {directory_to_store_experiment}")

    directory_to_store_definition = open_or_create_folder(directory_to_store_experiment, "definitions", create_new=False)
    print(f"Directory to store definition: {directory_to_store_definition}")

    directory_to_store_experiments = os.path.join(directory_to_store_experiment, "experiments")
    print(f"Directory to store experiments: {directory_to_store_experiments}")

    directory_to_store_logs = os.path.join(directory_to_store_experiment, "logs")
    print(f"Directory to store experiments: {directory_to_store_logs}")


    commands = []

    # for each model, setup a configuration to optimize with that model

    print()
    for model_index in range(len(model_paths)):

        model_name = model_names[model_index]
        model_path = model_paths[model_index]

        # SETUP CONFIGURATION TO OPTIMIZE

        fd = open(base_to_opt_config_path, 'r') 
        base_to_opt_config_str = fd.read() # reads the base configuration str
        fd.close()

        # modify configuration to use desired model
        base_to_opt_config_dict = dict_from_json_string(base_to_opt_config_str)
        rl_pipeline_input = base_to_opt_config_dict["input"]
        agents_input = rl_pipeline_input["agents_input"]
        policy_tuple = agents_input["policy"]
        policy_input = policy_tuple[1]

        policy_input["model"] = model_path

        # save changed configuration
        json_str_to_opt = json_string_of_component_dict(base_to_opt_config_dict)

        to_optimize_config_path = os.path.join(directory_to_store_definition, f'to_optimize_configuration_{model_name}.json')

        fd = open(to_optimize_config_path, 'w')
        fd.write(json_str_to_opt)
        fd.close()

        # directory where to store experiment
        to_store_experiment = os.path.join(directory_to_store_experiments, f'experiment_{model_name}')

        # SETUP EXPERIMENT COMMAND

        commands.append(
            ' '.join(
            [SCRIPT_PATH,
            "--LOGDIR", f'"{directory_to_store_logs}"',
            "--LOGBASENAME", f'"{model_name}"',
            "--EXPSTOREPATH", f'"{to_store_experiment}"',
            "--TOOPTIMIZECONFIG", f'"{to_optimize_config_path}"',
            "--CONFIG", f'"{hp_opt_config_path}"',
            "--RELPATH", f'"{model_name}"',
            ]
            )
        )

        print(f"Made command for model {model_name}:\n    {commands[len(commands) - 1]}\n")

    print()

    return commands
