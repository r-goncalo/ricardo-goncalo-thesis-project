from automl.component import InputSignature, requires_input_proccess

import torch

import random

from automl.ml.models.model_components import ModelComponent



from automl.rl.policy.policy import Policy

class StochasticPolicy(Policy):
    '''
    A policy wich selects actions given on probabilities
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
        self.initialize_model()
        
        
    def initialize_model(self):
    
        super().initialize_model()
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.model.pass_input({"input_shape" : self.model_input_shape, "output_shape" : self.model_output_shape})
        
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    @requires_input_proccess
    def predict(self, state):
        
        
        
        probabilitiesForActions : torch.Tensor = self.model.predict(state)
        
        dist = torch.distributions.Categorical(probabilitiesForActions)
        
        return dist.sample()
        
        
    
    @requires_input_proccess
    def random_prediction(self):
        return random.randint(0, self.model_output_shape.n - 1)