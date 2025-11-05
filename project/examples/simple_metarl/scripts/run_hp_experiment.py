
from automl.meta_rl.hp_optimization_pipeline import HyperparameterOptimizationPipeline
from automl.utils.json_utils.json_component_utils import gen_component_from_path
from automl.loggers.logger_component import DEBUG_LEVEL
from automl.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automl.basic_components.state_management import save_state
from automl.loggers.global_logger import activate_global_logger, get_global_level_artifact_directory


def main(hp_configuration_path='.\\configuration.json', to_optimize_configuration_path=None, path_to_store_experiment='.\\data\\experiments', num_trials=None, num_steps=None, sampler=None, create_new_directory=None, experiment_relative_path=None, global_logger_level=None):
    
    # the input for the hp optimization pipeline component
    hp_pipeline_input = {}
    
    if to_optimize_configuration_path != None:
        hp_pipeline_input["base_component_configuration_path"] = to_optimize_configuration_path
    
    hp_pipeline_input["base_directory"] = path_to_store_experiment

    if experiment_relative_path is not None:
        hp_pipeline_input["artifact_relative_directory"] = experiment_relative_path
    
    if num_steps != None:
        hp_pipeline_input["steps"] = num_steps
    
    if num_trials != None:
        hp_pipeline_input["n_trials"] = num_trials
    
    if create_new_directory != None:
        if isinstance(create_new_directory, str):
            create_new_directory = True if create_new_directory.lower() in ["true", "yes", "t", "y"] else False
        hp_pipeline_input["create_new_directory"] = create_new_directory
    
    if sampler != None:
        hp_pipeline_input["sampler"] = sampler
    
    # generate hp optimization pipeline component
    hp_optimization_pipeline : HyperparameterOptimizationPipeline = gen_component_from_path(hp_configuration_path)
    
    #pass the defined input
    hp_optimization_pipeline.pass_input(hp_pipeline_input)

    hp_optimization_pipeline.change_logger_level(DEBUG_LEVEL.INFO) # guarantees hp_optimization_pipeline has all its output
    
    if global_logger_level != None:
      
        activate_global_logger(hp_optimization_pipeline.get_artifact_directory(), global_logger_input={"necessary_logger_level" : global_logger_level})

        hp_optimization_pipeline.lg.writeLine(f"\nGLOBAL LOGGER ACTIVATED HERE WITH ARTIFACT DIRECTORY {get_global_level_artifact_directory()}\n")


    hp_optimization_pipeline.run()
    


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization pipeline.")
    
    parser.add_argument("--num_trials", type=int, default=None, help="Number of trials to run.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to run.")
    parser.add_argument("--sampler", type=str, default=None, help="Number of trials to run.")
    parser.add_argument("--create_new_directory", type=str, default=None, help="Number of trials to run.")

    parser.add_argument("--path_to_store_experiment", type=str, default='.\\data\\experiments', help="Directory to save results.")
    parser.add_argument("--experiment_relative_path", type=str, default=None, help="Relative directory to save results.")

    parser.add_argument("--hp_configuration_path", type=str, default='.\\configuration.json', help="Path to config of hp experiment.")
    parser.add_argument("--to_optimize_configuration_path", type=str, default='.\\to_optimize_configuration.json', help="Path to config to optimize")

    parser.add_argument("--global_logger_level", type=str, default=None, help="Path to config to optimize")

    args = parser.parse_args()

    main(num_trials=args.num_trials, num_steps=args.num_steps,
         create_new_directory=args.create_new_directory, sampler=args.sampler,
         path_to_store_experiment=args.path_to_store_experiment, 
         experiment_relative_path=args.experiment_relative_path,
         hp_configuration_path = args.hp_configuration_path,
         to_optimize_configuration_path = args.to_optimize_configuration_path,
         global_logger_level = args.global_logger_level
         )
    