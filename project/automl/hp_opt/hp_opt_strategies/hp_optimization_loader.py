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

MEMORY_REPORT_FILE = "memory_report.txt"

class HyperparameterOptimizationPipelineLoader(HyperparameterOptimizationPipeline):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         
                       }
            


    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
                
                
        self.trial_loaders : dict[str, StatefulComponentLoader] = {}
        
                
    
    # OPTIMIZATION -------------------------------------------------------------------------
    
    def _create_loader_for_component(self, trial : optuna.Trial, component_to_opt : Component_to_opt_type):

        self.lg.writeLine(f"Creating loader for trial {trial.number}")

        component_saver_loader = StatefulComponentLoader()
        component_saver_loader.define_component_to_save_load(component_to_opt)

        return component_saver_loader


    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
                
        component_to_opt = super()._create_component_to_optimize(trial)
        
        self.trial_loaders[trial.number] = self._create_loader_for_component(trial, component_to_opt)
                
        return component_to_opt
    
    
    def _load_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
                
        component_saver_loader : StatefulComponentLoader = self.trial_loaders[trial.number]
        component_to_opt : Component_to_opt_type = component_saver_loader.get_component()
        
        return component_to_opt    


    def get_component_to_test_path(self, trial : optuna.Trial) -> str:
        return self.get_component_to_test(trial).get_artifact_directory()


    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:

        if not trial.number in self.trial_loaders.keys():
            return self._create_component_to_optimize(trial)
        
        else:
            return self._load_component_to_test(trial)



    def _unload_component_to_test(self, trial : optuna.Trial):

        self.lg.writeLine(f"\nUnloading component to test for trial {trial.number} with current memory info\n: {torch.cuda.memory_summary() if torch.cuda.is_available() else 'No CUDA available'}", file=MEMORY_REPORT_FILE)
        
        component_saver_loader = self.trial_loaders[trial.number]
        component_saver_loader.unload_component()

        
    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''
        Called when a trial is over
        It is passed to optuna in the callbacks when the objective is defined
        '''        

        super().after_trial(study, trial)
                        
        self._unload_component_to_test(trial)
    
    
                    
    
    # EXPOSED METHODS -------------------------------------------------------------------------------------------------
                    
                    
    @requires_input_proccess
    def _algorithm(self): 

        self.lg.writeLine() 

        self.lg.writeLine(f"OPTIMIZING WITH {self.n_trials} TRIALS ------------------------------------------\n")      

        self.study.optimize( lambda trial : self.objective(trial), 
                       n_trials=self.n_trials,
                       callbacks=[self.after_trial])
        
        self.lg.writeLine(f"OPTIMIZATION WITH {self.n_trials} TRIALS OVER --------------------------------------------------------------------")

        try:
            self.lg.writeLine(f"Best parameters: {self.study.best_params}, used in trial {self.study.best_trial.number}, with best result {self.study.best_value}" )
            
        
        except Exception as e:
            self.lg.writeLine(f"Error getting best parameters: {e}")

