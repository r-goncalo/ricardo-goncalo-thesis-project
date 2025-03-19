

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
    def evaluate(self, component_to_evaluate : Component):
        pass
