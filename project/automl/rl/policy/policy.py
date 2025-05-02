from automl.component import Component, InputSignature

from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from


class Policy(Component):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''
        
    parameters_signature = {
        "model" : InputSignature(mandatory=False),
        "model_class" : InputSignature(mandatory=False),
        "model_input" : InputSignature(default_value={}),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature(),
    }   
    
    def proccess_input(self):
        
        super().proccess_input()
        self.initialize_model()
        
        
    def initialize_model(self):
        
        if not "model" in self.input.keys():
            self.model : ModelComponent = self.create_model()
        
        else:
            self.model : ModelComponent = self.input["model"]

            
        self.model.pass_input(self.input['model_input'])
        
        
        
    def create_model(self):
        
        if not "model_class" in self.input.keys():
            raise Exception("Model not defined and model class not defined")
        
        model_class = get_class_from(self.input["model_class"])
        
        return self.initialize_child_component(model_class)
        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        pass
    
    
    def random_prediction(self, state):
        pass
    

