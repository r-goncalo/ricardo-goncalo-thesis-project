import os
from typing import Union
from automl.component import InputSignature, Component, requires_input_proccess
from automl.basic_components.exec_component import ExecComponent
from automl.core.advanced_input_management import ComponentInputSignature
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline
from automl.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automl.hp_opt.optuna.custom_pruners import MixturePruner
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.result_logger import ResultLogger
from automl.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automl.loggers.logger_component import ComponentWithLogging

from automl.utils.files_utils import write_text_to_file
from automl.utils.json_utils.json_component_utils import gen_component_from_dict,  dict_from_json_string, json_string_of_component_dict, gen_component_from

import optuna

from automl.basic_components.state_management import StatefulComponent, StatefulComponentLoader
from automl.basic_components.seeded_component import SeededComponent
from automl.utils.configuration_component_utils import save_configuration
import torch

from automl.basic_components.state_management import save_state
 
import copy

class SimpleHyperparameterOptimizationPipeline(HyperparameterOptimizationPipeline):

    '''
    A simple Hyperparameter Optimization pipeline, which supports remembering a single configuration
    '''
    
    parameters_signature = {
                                              
                       }
            

    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
                
        self.component_being_optimized : Component_to_opt_type = None
                
    

    # OPTIMIZATION -------------------------------------------------------------------------
    
    def get_component_to_test_path(self, trial : optuna.Trial) -> str:

        if self.component_being_optimized is None:
            self.component_being_optimized = self._create_component_to_optimize(trial) 

        return self.component_being_optimized.get_artifact_directory()

    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        if self.component_being_optimized is None:
            self.component_being_optimized = self._create_component_to_optimize(trial) 
        
        return self.component_being_optimized
    
    def after_trial(self, study, trial):
        super().after_trial(study, trial)

        self.component_being_optimized = None

