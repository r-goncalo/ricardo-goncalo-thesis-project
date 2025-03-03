
from automl.component import Schema, requires_input_proccess
from automl.evaluator_component import EvaluatorComponent
from automl.rl.rl_pipeline import RLPipelineComponent


class RLPipelineEvaluator(EvaluatorComponent):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {}
    

    def proccess_input(self):
        
        super().proccess_input()
        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : RLPipelineComponent):
        pass