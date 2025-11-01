


from experiments import hp_experiments_sequence


def experiment_1(directory_to_store_experiment = "C:\\rgoncalo\\experiments",
                 base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json",
                 hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration_3.json", 
                 directory_of_models="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models",
                 experiment_name="sb3_zoo_dqn_cartpole_hp_opt_mult_samplers_pruners",
                 base_commands=None):


    directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = hp_experiments_sequence.setup_experiment_directories(
        directory_to_store_experiment=directory_to_store_experiment,
        experiment_name=experiment_name
    )

    # Basic parameters all commands will have
    base_command = {
            "create_new_directory" : "False",
            "path_to_store_experiment" : directory_to_store_experiments,
            "experiment_relative_path" : experiment_name
        }

    if base_commands == None:

        base_commands = [
            {
                **base_command,
                    "num_trials" : 50,
                    "sampler" : "Random", # this is to gain some knowledge first
                    "hp_configuration_path" : hp_opt_config_path,
                    "to_optimize_configuration_path" : base_to_opt_config_path,

             },

            {
                **base_command,
                    "num_trials" : 150,
                    "sampler": "TreeParzen"

            },
        ]

        print("\nBASE COMMANDS:\n")

        hp_experiments_sequence.print_commands(base_commands)

        print("END OF BASE COMMANDS\n")

    expanded_commands_for_models = hp_experiments_sequence.expand_commands_for_each_model(
        command_dicts=base_commands,
        directory_of_models=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions
    )

    hp_experiments_sequence.guarantee_same_path_in_commands(expanded_commands_for_models)

    return expanded_commands_strs_for_models


def experiment_2(directory_to_store_experiment = "C:\\rgoncalo\\experiments",
                 base_to_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\to_optimize_configuration.json",
                 hp_opt_config_path="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\configurations\\configuration_3.json", 
                 directory_of_models="C:\\rgoncalo\\experiment_definitions\\dqn_cartpole_sb3_zoo\\models",
                 experiment_name="sb3_zoo_dqn_cartpole_hp_opt_mult_samplers_pruners",
                 base_commands=None):


    directory_to_store_experiment, directory_to_store_definitions, directory_to_store_experiments, directory_to_store_logs = hp_experiments_sequence.setup_experiment_directories(
        directory_to_store_experiment=directory_to_store_experiment,
        experiment_name=experiment_name
    )

    # Basic parameters all commands will have
    base_command = {
            "create_new_directory" : "False",
            "path_to_store_experiment" : directory_to_store_experiments,
            "experiment_relative_path" : experiment_name
        }

    if base_commands == None:

        base_commands = [
            {
                **base_command,
                    "num_trials" : 50,
                    "sampler" : "Random", # this is to gain some knowledge first
                    "hp_configuration_path" : hp_opt_config_path,
                    "to_optimize_configuration_path" : base_to_opt_config_path,

             },

            {
                **base_command,
                    "num_trials" : 150,
                    "sampler": "TreeParzen"

            },
        ]

        print("\nBASE COMMANDS:\n")

        hp_experiments_sequence.print_commands(base_commands)

        print("END OF BASE COMMANDS\n")

    expanded_commands_for_models = hp_experiments_sequence.expand_commands_for_each_model(
        command_dicts=base_commands,
        directory_of_models=directory_of_models,
        directory_to_store_definitions=directory_to_store_definitions
    )

    hp_experiments_sequence.guarantee_same_path_in_commands(expanded_commands_for_models)

    expanded_commands_strs_for_models = hp_experiments_sequence.make_command_dicts_command_strings(expanded_commands_for_models)

    return expanded_commands_strs_for_models