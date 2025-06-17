from automl.component import InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
import torch

import random

from automl.ml.models.model_components import ModelComponent



from automl.rl.policy.policy import Policy

class StochasticPolicy(Policy):
    '''
    A policy wich selects actions given on probabilities
    '''
        
    parameters_signature = {
    }   
    
    def proccess_input(self):
        
        super().proccess_input()
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    @requires_input_proccess
    def predict(self, state):
        
        probabilitiesForActions : torch.Tensor = self.model.predict(state)
        
        dist = torch.distributions.Categorical(probabilitiesForActions)
        
        return dist.sample()
        
        
    
    @requires_input_proccess
    def random_prediction(self):
        return random.randint(0, self.model_output_shape.n - 1)