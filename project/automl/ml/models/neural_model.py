import os
from automl.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn
import torch.nn.functional as F    

from automl.component import Component, InputSignature, requires_input_proccess
import random

from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space


class FullyConnectedModelSchema(TorchModelComponent):
    
    
    '''
        Represents a fully connected neural network model schema.
        
        The class "Model_Class" is the actual model architecture, which is a subclass of nn.Module
        A class that extends this schema could reimplement the Model_Class to define the architecture of the model.
    '''
        
    # The actual model architecture
    class Model_Class(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, hidden_layers):
            super(FullyConnectedModelSchema.Model_Class, self).__init__()
            
            self.input_size = input_size
            
            layers = []
            prev_size = input_size
            
            for _ in range(hidden_layers):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            
            layers.append(nn.Linear(hidden_size, output_size))
            
            self.network = nn.Sequential(*layers)

        def forward(self, x : torch.Tensor):

            if isinstance(x, torch.Tensor):
                x = x.view(-1, self.input_size) #the x is reshaped so it has 2 dimensions, the first one is the batch and the second the input size 
                        
            return self.network(x)
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "hidden_layers" : InputSignature(description="Number of hidden layers"),
        "hidden_size": InputSignature(description="Size of hidden layers"),
        "device": InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }    
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        

            
    def _setup_values(self):
        super()._setup_values()    

        self.input_size: int =  discrete_input_layer_size_of_space(self.input_shape)
        
        self.hidden_size: int = self.input["hidden_size"]
        self.hidden_layers: int = self.input["hidden_layers"]
        
        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)
                       

        
    def _initialize_model(self):

        self.model : nn.Module = type(self).Model_Class(
            input_size=self.input_size,
                hidden_size=self.hidden_size, 
                output_size=self.output_size,
                hidden_layers=self.hidden_layers
            )
        
    def _is_model_well_formed(self):
        super()._is_model_well_formed()
        
        # TODO: verify if size and so on are coherent
        
                            
    # EXPOSED METHODS --------------------------------------------