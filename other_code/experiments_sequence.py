


import os
import shutil

from automl.utils.files_utils import open_or_create_folder
from automl.utils.json_component_utils import dict_from_json_string, json_string_of_component_dict


SCRIPT_PATH = "C:\\rgoncalo\\ricardo-goncalo-thesis-project\\other_code\\RunSimplyHpExpCommand.bat"

BASE_COMMAND = [
    SCRIPT_PATH,
    "--COMMAND",
    "python",
    "C:\\rgoncalo\\ricardo-goncalo-thesis-project\\project\\examples\\simple_metarl\\scripts\\run_hp_experiment.py"
]

def hp_opt_command_sequence(
                                 parameter_dict : dict,
                            ):
    
    command_args_list = [*BASE_COMMAND]

    for key, value in parameter_dict.items():
        command_args_list.append(f"--{key}")
        command_args_list.append(value)

    return command_args_list


def hp_opt_for_models(directory_of_models,
                                 directory_to_store_experiment,
                                 base_to_opt_config_path, 
                                 hp_opt_config_path,
                                 parameter_dict_list  : list [dict] = [{}],
                                 experiment_name = "hp_opt_for_models",
                                 models_to_test=None):
    
    print("\nCREATING COMMAND SEQUENCES FOR MODELS ------------------------------------------------")
    
    print(f"Base configuration to optimize path: {base_to_opt_config_path}")

    print(f"Base configuration of Hyperparameter Optimization pipeline: {hp_opt_config_path}")

    print(f"\nDirectory of models: {directory_of_models}")

    if models_to_test == None:
        # get model paths
        model_paths = [
            os.path.join(directory_of_models, f)
            for f in os.listdir(directory_of_models)
        ]

    else:
        print(f"Models to test: {models_to_test}")
        model_paths = [
            os.path.join(directory_of_models, f)
            for f in os.listdir(directory_of_models)
            if f in models_to_test
        ]   

    model_names = [os.path.basename(path) for path in model_paths]

    print(f"Models found: {model_names}")

    directory_to_store_experiment = open_or_create_folder(directory_to_store_experiment, folder_name=f"{experiment_name}")
    print(f"\nDirectory to store experiment: {directory_to_store_experiment}")

    directory_to_store_definition = open_or_create_folder(directory_to_store_experiment, "definitions", create_new=False)
    print(f"Directory to store definition: {directory_to_store_definition}")

    directory_to_store_experiments = os.path.join(directory_to_store_experiment, "experiments")
    print(f"Directory to store experiments: {directory_to_store_experiments}")

    directory_to_store_logs = os.path.join(directory_to_store_experiment, "logs")
    print(f"Directory to store experiments: {directory_to_store_logs}")


    command_sequence_list = []

    # for each model, setup a configuration to optimize with that model

    print()
    for model_index in range(len(model_paths)):

        model_name = model_names[model_index]
        model_path = model_paths[model_index]

        print(f"    Dealing with model {model_name} in {model_path}")

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

        print(f"    Model will have its optimization config in {to_optimize_config_path}")

        fd = open(to_optimize_config_path, 'w')
        fd.write(json_str_to_opt)
        fd.close()

        # SETUP EXPERIMENT COMMAND SEQUENCE

        base_parameter_dict = {
            "create_new_directory" : "False",
            "path_to_store_experiment" : directory_to_store_experiments,
            "hp_configuration_path" : hp_opt_config_path,
            "to_optimize_configuration_path" : to_optimize_config_path,
            "experiment_relative_path" : model_name
        }

        print(f"    Creating commands for HP optimization...\n")

        command_sequence = []

        for parameter_dict in parameter_dict_list:

            command_to_add = hp_opt_command_sequence({**parameter_dict, **base_parameter_dict})
            
            print(f"    For parameter dict {parameter_dict}, creating command:\n        {command_to_add}\n")

            command_sequence.append(
                ' '.join([str(command_arg) for command_arg in command_to_add]) # to make it a str
            )

        command_sequence_list.append(command_sequence)


    print("\nEND CREATION OF COMMAND SEQUENCES FOR MODELS ----------------------------------------------\n")

    return command_sequence_list

