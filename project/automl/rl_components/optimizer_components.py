from ..component import Component
from ..component import input_signature
import torch.optim as optim
import torch.nn
import numpy as nn

from abc import abstractmethod

class OptimizerComponent(Component):
    
    input_signature = {**Component.input_signature}
    
    @abstractmethod
    def optimize_model(self, predicted, correct) -> None:
        
        '''
            Optimizes the model based on the prediction(s) and correct value(s)
            
            Args:
                predicted is the prediced value(s) of the model for a given input
                correct is the correct value(s) for the given input
                
        '''
        
        pass

class AdamOptimizer(OptimizerComponent):

    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = {**OptimizerComponent.input_signature, 
                       "torch_params" : input_signature(),
                       "learning_rate" : input_signature(),
                       "amsgrad" : input_signature(default_value=True)}    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.torch_adam_opt = optim.AdamW(params=self.input["torch_params"],lr=self.input["learning_rate"], amsgrad=self.input["amsgrad"])

    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    def optimize_model(self, predicted, correct) -> None:
        
        super().optimize_model(self, predicted, correct)
        
        # Compute Huber loss TODO : The loss should not be hard calculated like this
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted, correct)
        
        #Optimize the model
        self.optimizer.zero_grad() #clears previous accumulated gradients in the optimizer
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) #clips gradients to prevent them from reaching values over 100, trying to resolve explosive gradients problem
        self.optimizer.step()