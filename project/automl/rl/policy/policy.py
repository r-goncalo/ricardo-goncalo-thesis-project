from automl.component import Component, InputSignature

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from


class Policy(Component):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''
        
    parameters_signature = {
        
        "model" : ComponentInputSignature(),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature(),
    }   
    
    def proccess_input(self):
        
        super().proccess_input()
        
        self.model : ModelComponent = ComponentInputSignature.get_component_from_input(self, "model")
        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        pass
    
    
    def random_prediction(self, state):
        pass
    

