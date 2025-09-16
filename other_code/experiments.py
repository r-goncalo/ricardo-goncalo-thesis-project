


import os


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


def sb3_montaincar_semi_trained_2(directory_of_models, hp_opt_config_path):

    # get model paths
    model_paths = [
        os.path.join(directory_of_models, f)
        for f in os.listdir(directory_of_models)
        if os.path.isfile(os.path.join(directory_of_models, f))
    ]

    model_names = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]


    # for each model, setup a configuration to optimize with that model


    to_return = []

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
