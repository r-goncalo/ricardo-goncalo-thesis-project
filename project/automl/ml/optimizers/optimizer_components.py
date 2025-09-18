from automl.component import Component, InputSignature, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.model_components import ModelComponent
import torch.optim as optim
import torch.nn as nn

from abc import abstractmethod

class OptimizerSchema(Component):
    
    parameters_signature = {"model" : ComponentInputSignature(ignore_at_serialization=True)}
    
    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.model : ModelComponent = ComponentInputSignature.get_component_from_input(self, "model")

        
        
        
    @abstractmethod
    def optimize_model(self, predicted, correct) -> None:
        
        '''
            Optimizes the model based on the prediction(s) and correct value(s)
            
            Args:
                predicted is the prediced value(s) of the model for a given input
                correct is the correct value(s) for the given input
                
        '''
        
        pass

class AdamOptimizer(OptimizerSchema):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                       "learning_rate" : InputSignature(default_value=0.001),
                       "amsgrad" : InputSignature(default_value=True),
                       "clip_grad_value" : InputSignature(mandatory=False, description="If defined, it clips the gradients to the given value"),
                       "clip_grad_norm" : InputSignature(mandatory=False, description="If defined, it clips the gradients to the given norm")
                       }    
    
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.params = self.model.get_model_params() #gets the model parameters to optimize
                
        self.torch_adam_opt = optim.AdamW(params=self.params,lr=self.input["learning_rate"], amsgrad=self.input["amsgrad"])

        self.clip_grad_value = None
        if "clip_grad_value" in self.input:
            self.clip_grad_value = self.input["clip_grad_value"]
        
        self.clip_grad_norm = None
        if "clip_grad_norm" in self.input:  
            self.clip_grad_norm = self.input["clip_grad_norm"]

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
        if self.clip_grad_value is not None:
            nn.utils.clip_grad_value_(self.params, self.clip_grad_value)
        
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        
        self.torch_adam_opt.step()
        
        

class SimpleSGDOptimizer(OptimizerSchema):

    parameters_signature = {
        "learning_rate": InputSignature(default_value=0.01)
    }

    def proccess_input_internal(self):
        super().proccess_input_internal()

        # Get model parameters
        self.params = self.model.get_model_params()

        # Define SGD optimizer
        self.sgd_optimizer = optim.SGD(self.params, lr=self.input["learning_rate"])

    @requires_input_proccess
    def optimize_model(self, predicted, correct) -> None:
        super().optimize_model(predicted, correct)

        # Use simple Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(predicted, correct)

        # Zero previous gradients
        self.sgd_optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Optional gradient clipping to prevent instability
        nn.utils.clip_grad_norm_(self.params, max_norm=10)

        # Update model parameters
        self.sgd_optimizer.step()