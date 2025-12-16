from automl.component import  InputSignature, requires_input_proccess


import torch

import random

from automl.rl.policy.policy import Policy
from automl.utils.shapes_util import single_action_shape


class QPolicy(Policy):
    '''
    It uses a model to calculate the Q action values of each action and calculates the action as such
    '''
        
    parameters_signature = {
        
    }   
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()       

        
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    @requires_input_proccess
    def predict(self, state):
            
        valuesForActions : torch.Tensor = self.model.predict(state) #a tensor ether in the form of [q values for each action] or [[q value for each action]]?
        
        #tensor of max values and tensor of indexes
        _, max_indexes = valuesForActions.max(dim=1)
                        
        return max_indexes
    
    
    @requires_input_proccess
    def random_prediction(self):    
        return torch.randint(
            0, #low
            int(self.policy_output_shape), #high
            (1,), #size
            device=self.device
        )