from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from

from automl.utils.shapes_util import single_action_shape


class PolicyInterface(Component):
    
    '''
    A policy interface, defining only the methods necessary for a policy
    This abstracts wrappers and our own implemented strategies
    '''
        
    def get_policy_shape(self):
        raise NotImplementedError()
        
    def predict(self, state):
        raise NotImplementedError()
    
    def random_prediction(self, state):
        raise NotImplementedError()



class Policy(PolicyInterface):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''
        
    parameters_signature = {
        
        "model" : ComponentInputSignature(),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature(),
        "device" : InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }   
    
    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.model : ModelComponent = ComponentInputSignature.get_component_from_input(self, "model")
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.policy_output_shape = single_action_shape(self.model_output_shape)
        
        self.device = self.input["device"]
                
        self.model.pass_input({"input_shape" : self.model_input_shape, "output_shape" : self.model_output_shape, "device" : self.device}) 
        
    
    @requires_input_proccess
    def get_policy_shape(self):
        return self.policy_output_shape
        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        pass
    
    
    def random_prediction(self, state):
        pass
    

