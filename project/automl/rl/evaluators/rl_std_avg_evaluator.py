
from automl.component import Schema, requires_input_proccess
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.loggers.result_logger import ResultLogger

from automl.core.input_management import InputSignature

class RLPipelineAvgStdEvaluator(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {
        "n_results_to_use" : InputSignature(default_value=10)
    }
    

    def proccess_input(self):
        
        super().proccess_input()
        
        self.n_results_to_use = self.input["n_results_to_use"]
        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : RLPipelineComponent):
        
        results_logger : ResultLogger = component_to_evaluate.get_results_logger() 

        avg_result, std_result = results_logger.get_avg_and_std_n_last_results(10, 'total_reward')

        result = avg_result - (std_result / 4)
        
        return result
        