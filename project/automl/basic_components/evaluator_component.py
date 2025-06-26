

from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature

from abc import abstractmethod

class EvaluatorComponent(Component):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
                    }    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()


    # EVALUATION -------------------------------------------------------------------------------

    
    @requires_input_proccess
    @abstractmethod
    def get_metrics_strings(self) -> list[str]:
        return []
    
    @requires_input_proccess
    @abstractmethod
    def evaluate(self, component_to_evaluate : Component) -> dict:
        '''
        Returns a dictionary with the results of the evaluation
        A value for the key "result" will always exist
        '''
        return {}


class ComponentWithEvaluator(Component):
    
    '''
    A component that has an evaluator
    It is used to evaluate the component itself
    '''
    
    parameters_signature = {
        "component_evaluator" : InputSignature(mandatory=True),
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.last_evaluation = {}
    
    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.component_evaluator : EvaluatorComponent = self.input["component_evaluator"]
        
    
    def evaluate_this_component(self) -> dict:
        
        '''
        Evaluates this component using its evaluator
        '''
        self.last_evaluation = self.component_evaluator.evaluate(self)
        
        return self.last_evaluation
    
    def get_last_evaluation(self):
        return self.last_evaluation