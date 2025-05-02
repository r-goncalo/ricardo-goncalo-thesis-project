from automl.component import  InputSignature, requires_input_proccess


import torch

import random

from automl.rl.policy.policy import Policy

class QPolicy(Policy):
    '''
    It uses a model to calculate the Q action values of each action and calculates the action as such
    '''
        
    parameters_signature = {
        "model" : InputSignature(mandatory=False),
        "model_class" : InputSignature(mandatory=False),
        "model_input" : InputSignature(default_value={}),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature()
    }   
    
    def proccess_input(self):
        
        super().proccess_input()        
        
    def initialize_model(self):
    
        super().initialize_model()
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.model.pass_input({"input_shape" : self.model_input_shape, "output_shape" : self.model_output_shape})
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    @requires_input_proccess
    def predict(self, state):
            
        valuesForActions : torch.Tensor = self.model.predict(state) #a tensor ether in the form of [q values for each action] or [[q value for each action]]?
        
        #tensor of max values and tensor of indexes
        max_values, max_indexes = valuesForActions.max(dim=1)
        
        return max_indexes
    
    
    @requires_input_proccess
    def random_prediction(self):
        return random.randint(0, self.model_output_shape.n - 1)