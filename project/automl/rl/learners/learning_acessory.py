from automl.basic_components.dynamic_value import get_value_or_dynamic_value
from automl.component import Component

from automl.core.advanced_input_management import ComponentInputSignature

class LearningAcessory(Component):
    
    '''
    Executes some proccess for a learner
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "learner" : ComponentInputSignature()
    }    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.learner = self.get_input_value("learner")
                
            
    def pre_learning(self):
        pass


    def post_learning(self, values : dict):
        pass




