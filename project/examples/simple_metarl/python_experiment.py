import sys
import os

sys.path.append(os.path.abspath("../..")) #make the folder "automl" part of this

from automl.meta_rl.hyperparameter_suggestion import HyperparameterSuggestion
from automl.base_configurations.base_configurations import load_configuration_dict
import optuna

from automl.meta_rl.hp_optimization_pipeline import HyperparameterOptimizationPipeline


if __name__ == '__main__':

    hyperparameters_to_change : list[HyperparameterSuggestion] = []

    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='num_episodes', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["num_episodes"])
                                    ],
                                    #value_suggestion = ('int', {'low':200, 'high':800, 'step':100}) 
                                    value_suggestion = ('int', {'low':6000, 'high':6000}) 
                                ),
                                ]       

    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                HyperparameterSuggestion(
                                    name='discount_factor', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "discount_factor"])
                                    ],
                                    value_suggestion = ('float', {'low':0.5, 'high':0.99}) 
                                )
                                ]      


    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='epsilon_start', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "exploration_strategy_input", "epsilon_start"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.999}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_end', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "exploration_strategy_input", "epsilon_end"])
                                    ],
                                    value_suggestion = ('float', {'low':0.05, 'high':0.3}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_decay', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "exploration_strategy_input", "epsilon_decay"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.9999}) 
                                 )

    ]

    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='hidden_layers', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "model_input", "hidden_layers"])
                                    ],
                                    value_suggestion = ('int', {'low':2, 'high':8}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='hidden_size', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "model_input", "hidden_size"])
                                    ],
                                    value_suggestion = ('cat', {'choices' : [16, 32, 64, 128, 256]}) 
                                )
    ]

    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='target_update_rate', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "learner_input", "target_update_rate"])
                                    ],
                                    value_suggestion = ('float', {'low':0.01, 'high':0.15}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='learning_rate', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "learner_input", "optimizer_input", "learning_rate"])
                                    ],
                                    value_suggestion = ('float', {'low':0.000001, 'high':0.1}) 
                                )            
    ]

    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='memory_capacity', 
                                    hyperparameter_localizations= [
                                        ('RLPipelineComponent', ["agents_input", "memory_input", "capacity"])
                                    ],
                                    value_suggestion = ('int', {'low':100, 'high':1000}) 
                                )

    ]

    hp_optimization_input = {
        "configuration_dict" : load_configuration_dict('basic_rl'),
        "hyperparameters_range_list" : hyperparameters_to_change,
        "n_trials" : 50,
        "steps" : 3,
        "pruner" : optuna.pruners.PercentilePruner(percentile=35.0)
        }


    hp_optimization_pipeline = HyperparameterOptimizationPipeline(hp_optimization_input)

    hp_optimization_pipeline.run()

    hp_optimization_pipeline.save_configuration()