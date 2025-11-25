from automl.basic_components.component_group import RunnableComponentGroup
from automl.component import InputSignature, Component, requires_input_proccess
from automl.basic_components.artifact_management import ArtifactComponent
from automl.basic_components.exec_component import ExecComponent
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.result_logger import ResultLogger
from automl.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline, Component_to_opt_type
from automl.rl.rl_pipeline import RLPipelineComponent

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automl.utils.json_utils.json_component_utils import gen_component_from_dict,  dict_from_json_string

import optuna

from automl.hp_opt.hyperparameter_suggestion import HyperparameterSuggestion

from automl.utils.random_utils import generate_and_setup_a_seed

from automl.basic_components.state_management import StatefulComponent, StatefulComponentLoader

from automl.basic_components.whole_configurations import component_group as BaseComponentGroupConfig 

class HyperparameterOptimizationPipelineWithGroups(HyperparameterOptimizationPipeline):
    
    parameters_signature = {
        
                        "number_of_opt_per_group" : InputSignature(default_value=3)
                                                    
                       }
            
    # INITIALIZATION -----------------------------------------------------------------------------

    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
                
        self.number_of_opt_per_group = self.get_input_value("number_of_opt_per_group")
        

    def _create_component_to_optimize_configuration(self, trial : optuna.Trial) -> dict:
        
        '''Returns the configuration of the group of components that are being optimized with this hyperparameter configuration'''
        
        config_of_opt = super()._create_component_to_optimize_configuration(trial)
        
        config_of_opt = BaseComponentGroupConfig.config_dict(config_of_opt, self.evaluator_component, self.number_of_opt_per_group)
        
        return config_of_opt
        
    
        
    def _try_evaluate_component(self, component_to_test : Component_to_opt_type | RunnableComponentGroup) -> float:
        
        if self.evaluator_component is None:
            raise NotImplementedError("The evaluator component is not set. Please set an evaluator component before evaluating the component.")
        
        
            
        
        return super()._try_evaluate_component(component_to_test)
        

        
        
        
