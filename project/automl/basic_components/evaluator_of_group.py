

from automl.basic_components.component_group import RunnableComponentGroup
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.result_logger import ResultLogger
import pandas



class EvaluatorComponentOfGroup(EvaluatorComponent):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
        "std_deviation_factor" : InputSignature(default_value=4, description="The factor to be used to calculate the standard deviation")
    }

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.std_deviation_factor = self.input["std_deviation_factor"]

        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : RunnableComponentGroup) -> dict:
        '''
        Evaluates a component group
        '''
        if not isinstance(component_to_evaluate, RunnableComponentGroup):
            raise TypeError("The component to evaluate must be a RunnableComponentGroup")
        
        results_lg_of_group : ResultLogger = component_to_evaluate.get_aggregate_results_lg()
        
        dataframe_of_group : pandas.DataFrame = results_lg_of_group.get_dataframe()
        
        max_times_ran = dataframe_of_group["times_ran"].max()

        # Filter rows where times_ran == max_times_ran
        df_max_times = dataframe_of_group[dataframe_of_group["times_ran"] == max_times_ran]

        # For each component_index, keep the row with max_times_ran
        df_max_grouped = df_max_times.groupby("component_index").first().reset_index()

        mean_result = df_max_grouped["result"].mean()
        std_result = df_max_grouped["result"].std()
        
        result = mean_result - (std_result / self.std_deviation_factor)
        
        return {"result" : result}
        
