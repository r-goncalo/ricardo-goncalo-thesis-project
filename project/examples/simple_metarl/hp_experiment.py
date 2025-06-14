from automl.meta_rl.hyperparameter_suggestion import HyperparameterSuggestion
from automl.base_configurations.base_configurations import load_configuration_dict
import optuna
from automl.meta_rl.hp_optimization_pipeline import HyperparameterOptimizationPipeline




def get_hyperparameters_to_change() -> list[HyperparameterSuggestion]:
    
    hyperparameters_to_change : list[HyperparameterSuggestion] = [] 
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                                
                            HyperparameterSuggestion(
                                name='discount_factor', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "discount_factor"])
                                ],
                                value_suggestion = ('float', {'low':0.5, 'high':0.99}) 
                            )
                            ]       
    
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='epsilon_start', 
                                    hyperparameter_localizations= [
                                        ([], ["agents_input", "exploration_strategy_input", "epsilon_start"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.999}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_end', 
                                    hyperparameter_localizations= [
                                        ([], ["agents_input", "exploration_strategy_input", "epsilon_end"])
                                    ],
                                    value_suggestion = ('float', {'low':0.05, 'high':0.3}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_decay', 
                                    hyperparameter_localizations= [
                                        ([], ["agents_input", "exploration_strategy_input", "epsilon_decay"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.9999}) 
                                 )

    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='hidden_layers', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "policy_input", "model_input", "hidden_layers"])
                                ],
                                value_suggestion = ('int', {'low':2, 'high':8}) 
                            ),
                             HyperparameterSuggestion(
                                name='hidden_size', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "policy_input", "model_input", "hidden_size"])
                                ],
                                value_suggestion = ('cat', {'choices' : [16, 32, 64, 128, 256]}) 
                            )
    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='target_update_rate', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "learner_input", "target_update_rate"])
                                ],
                                value_suggestion = ('float', {'low':0.01, 'high':0.15}) 
                            ),
                             HyperparameterSuggestion(
                                name='learning_rate', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "learner_input", "optimizer_input", "learning_rate"])
                                ],
                                value_suggestion = ('float', {'low':0.000001, 'high':0.1}) 
                            )            
    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='memory_capacity', 
                                hyperparameter_localizations= [
                                    ([], ["agents_input", "memory_input", "capacity"])
                                ],
                                value_suggestion = ('int', {'low':100, 'high':1000}) 
                            )
                           
    
    ]
    
    return hyperparameters_to_change

def get_configuration_dict(*args, **kwargs):
    
    return load_configuration_dict('basic_dqn', *args, **kwargs)


def gen_hp_optimization_input(hyperparameters_to_change, configuration_dict, num_trials=20):
    
    return {
    "configuration_dict" : configuration_dict,
    "hyperparameters_range_list" : hyperparameters_to_change,
    "n_trials" : num_trials,
    "steps" : 2,
    "pruner" : optuna.pruners.PercentilePruner(percentile=25.0),
    "base_directory" :  'data\\experiments'
    }
    
    
    
def main(num_episodes, num_trials):
    
    hyperparameters_to_change = get_hyperparameters_to_change()
    
    print("Generating hyperparameters to change")

    configuration_dict = get_configuration_dict(num_episodes=num_episodes)

    hp_opt_input = gen_hp_optimization_input(hyperparameters_to_change, configuration_dict, num_trials)
    
    hp_opt_pipeline = HyperparameterOptimizationPipeline(hp_opt_input)
    
    hp_opt_pipeline.run()
    
    #hp_opt_pipeline.save_configuration()
    
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization pipeline.")
    parser.add_argument("--num_episodes", type=int, default=12, help="Number of episodes to run.")
    parser.add_argument("--num_trials", type=int, default=20, help="Number of trials to run.")

    args = parser.parse_args()

    main(num_episodes=args.num_episodes, num_trials=args.num_trials)