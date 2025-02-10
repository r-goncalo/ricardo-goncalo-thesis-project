from ..component import Component, InputSignature, requires_input_proccess
import torch.optim as optim
import torch.nn as nn

from abc import abstractmethod

class OptimizerComponent(Component):
        
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

    parameters_signature = {"model_params" : InputSignature(),
                       "learning_rate" : InputSignature(default_value=0.001),
                       "amsgrad" : InputSignature(default_value=True)}    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.params = self.input["model_params"]
                
        self.torch_adam_opt = optim.AdamW(params=self.params,lr=self.input["learning_rate"], amsgrad=self.input["amsgrad"])

    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    @requires_input_proccess
    def optimize_model(self, predicted, correct) -> None:
        
        super().optimize_model(predicted, correct)
        
        # Compute Huber loss TODO : The loss should not be hard calculated like this
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted, correct)
        
        #Optimize the model
        self.torch_adam_opt.zero_grad() #clears previous accumulated gradients in the optimizer
        loss.backward()
        
        # In-place gradient clipping
        nn.utils.clip_grad_value_(self.params, 100) #clips gradients to prevent them from reaching values over 100, trying to resolve explosive gradients problem
        self.torch_adam_opt.step()