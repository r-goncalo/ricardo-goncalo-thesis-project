


import os

from project.automl.utils.files_utils import open_or_create_folder
from project.automl.utils.json_component_utils import dict_from_json_string, json_string_of_component_dict


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

    # get model paths
    model_paths = [
        os.path.join(directory_of_models, f)
        for f in os.listdir(directory_of_models)
        if os.path.isfile(os.path.join(directory_of_models, f))
    ]

    model_names = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]

    directory_to_store_experiment = open_or_create_folder(directory_to_store_experiment)

    directory_to_store_definition = open_or_create_folder(directory_to_store_experiment, "definitions", create_new=False)

    directory_to_store_experiments = open_or_create_folder(directory_to_store_experiment, "experiments", create_new=False)

    commands = []

    # for each model, setup a configuration to optimize with that model

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
            f'{SCRIPT_PATH} --LOGBASENAME {model_name} --EXPSTOREPATH {to_store_experiment} --TOOPTIMIZECONFIG {to_optimize_config_path} --CONFIG {hp_opt_config_path}',

        )
