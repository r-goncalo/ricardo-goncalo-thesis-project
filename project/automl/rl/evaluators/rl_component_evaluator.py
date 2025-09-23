
from automl.component import Component, requires_input_proccess
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.rl.rl_pipeline import RLPipelineComponent


class RLPipelineEvaluator(EvaluatorComponent):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {}
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):
        return super()._evaluate(component_to_evaluate)