

import pandas
from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.core.input_management import InputSignature

from abc import abstractmethod

from automl.loggers.logger_component import ComponentWithLogging

class EvaluatorComponent(Component):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
                    }    
    
    exposed_values = {
        
        "last_evaluation" : {}
    
    }

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()


    # EVALUATION -------------------------------------------------------------------------------

    
    @requires_input_proccess # needs to be extended
    def get_metrics_strings(self) -> list[str]:
        return []
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : Component) -> dict:
        '''
        Returns a dictionary with the results of the evaluation
        A value for the key "result" will always exist
        '''
        results = self._evaluate(component_to_evaluate)

        self.values["last_evaluation"] = results

        return results

    #needs to be extended
    def _evaluate(self, component_to_evaluate : Component) -> dict:
        return {}


class ComponentWithEvaluator(Component):
    
    '''
    A component that has an evaluator
    It is used to evaluate the component itself
    '''
    
    parameters_signature = {
        "component_evaluator" : ComponentInputSignature(mandatory=False),
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.last_evaluation = {}
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        if "component_evaluator" not in self.input.keys():
            self.component_evaluator = None
        
        else:
            self.component_evaluator : EvaluatorComponent = ComponentInputSignature.get_value_from_input(self, "component_evaluator")        
    
    def evaluate_this_component(self) -> dict:
        
        '''
        Evaluates this component using its evaluator
        '''
        
        if self.component_evaluator is None:
            raise Exception("This component does not have an evaluator")
        
        self.last_evaluation = self.component_evaluator.evaluate(self)
        
        return self.last_evaluation
    
    def get_last_evaluation(self):
        
        if self.component_evaluator is None:
            raise Exception("This component does not have an evaluator")
        
        elif not self.last_evaluation:
            raise Exception("This component has not been evaluated yet")
        
        return self.last_evaluation