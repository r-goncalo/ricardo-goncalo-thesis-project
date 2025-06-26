
from automl.component import Component, requires_input_proccess
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.loggers.result_logger import ResultLogger

from automl.core.input_management import InputSignature

class LastValuesAvgStdEvaluator(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines, which evaluates the last results it has and uses them to compute a result, penalizing high variance and using the mean as the base value
    
    This is meant to be used not as a final evaluation of a component, but as an intermediary evaluator at training time.
    
    '''
    
    parameters_signature = {
        "n_results_to_use" : InputSignature(default_value=10),
        "std_deviation_factor" : InputSignature(default_value=4, description="The factor to be used to calculate the standard deviation")
    }
    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.n_results_to_use = self.input["n_results_to_use"]
        self.std_deviation_factor = self.input["std_deviation_factor"]
        


    # EVALUATION -------------------------------------------------------------------------------
    
    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), "result"]
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : RLPipelineComponent):
        
        results_logger : ResultLogger = component_to_evaluate.get_results_logger() 
        
        n_results_to_use = self.n_results_to_use
        n_rows = results_logger.get_number_of_rows()
        
        if n_results_to_use > n_rows:
            n_results_to_use = n_rows

        avg_result, std_result = results_logger.get_avg_and_std_n_last_results(n_results_to_use, 'total_reward')

        result = avg_result - (std_result / self.std_deviation_factor)
        
        return {"result" : result, **super().evaluate(component_to_evaluate)}
        