

from automl.component import Component, requires_input_proccess



class EvaluatorComponent(Component):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
                    }    

    def proccess_input(self):
        
        super().proccess_input()


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def get_metrics_strings(self) -> list[str]:
        pass
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : Component) -> dict:
        '''
        Returns a dictionary with the results of the evaluation
        A value for the key "result" will always exist
        '''
        pass


class ComponentWithEvaluator(Component):
    
    '''
    A component that has an evaluator
    It is used to evaluate the component itself
    '''
    
    parameters_signature = {
        "component_evaluator" : EvaluatorComponent,
    }
    
    def proccess_input(self):
        
        super().proccess_input()
        
        self.component_evaluator : EvaluatorComponent = self.input["component_evaluator"]
        
    
    def evaluate_this_component(self) -> dict:
        
        '''
        Evaluates this component using its evaluator
        '''
        return self.component_evaluator.evaluate(self)