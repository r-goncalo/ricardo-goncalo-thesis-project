import os
from automl.basic_components.state_management import StatefulComponent
import torch
import torch.nn as nn
import torch.nn.functional as F    

from automl.component import Component, InputSignature, requires_input_proccess
import random

from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space

from automl.ml.models.model_components import ModelComponent

class FullyConnectedModelSchema(ModelComponent, StatefulComponent):
    
    
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
                #print(x.shape)
                x = x.view(-1, self.input_size) #the x is reshaped so it has 2 dimensions, the first one is the batch and the second the input size 
                        
            return self.network(x)
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "hidden_layers" : InputSignature(description="Number of hidden layers"),
        "hidden_size": InputSignature(description="Size of hidden layers"),
        "device": InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }    
    
    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.input_size: int =  discrete_input_layer_size_of_space(self.input_shape)
        
        self.hidden_size: int = self.input["hidden_size"]
        self.hidden_layers: int = self.input["hidden_layers"]
        
        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)
                
        self.initialize_model()
        
        if "device" in self.input.keys():
            self.model.to(self.input["device"])
            
        
    def initialize_model(self):
        
        if not hasattr(self, "model") or self.model is None: #if the model is not already loaded
            
            self.model = type(self).Model_Class(
                input_size=self.input_size, 
                hidden_size=self.hidden_size, 
                output_size=self.output_size,
                hidden_layers=self.hidden_layers
            )
            
        else: #the model could be already loaded from a state, in which case we just need to check if the input and output sizes are compatible
            
            #TODO: check if the model is compatible with the input and output sizes
            pass
        
                            
    # EXPOSED METHODS --------------------------------------------
    
    @requires_input_proccess
    def get_model_params(self):
        '''returns a list of model parameters'''
        return list(self.model.parameters())
    
    @requires_input_proccess
    def predict(self, state):
        super().predict(state)
        return self.model(state)

    
    @requires_input_proccess            
    def update_model_with_target(self, target_model, target_model_weight):
        
        '''
        Updates the weights of this model with the weights of the target model
        
        @param target_model_weight is the relevance of the target model, 1 will mean a total copy, 0 will do nothing, 0.5 will be an average between the models
        '''
        
        with torch.no_grad():
            this_model_state_dict = self.model.state_dict()
            target_model_state_dict = target_model.model.state_dict()

            for key in target_model_state_dict:
                this_model_state_dict[key] = (
                    target_model_state_dict[key] * target_model_weight + 
                    this_model_state_dict[key] * (1 - target_model_weight)
                )
            
            self.model.load_state_dict(this_model_state_dict)
    
    # UTIL -----------------------------------------------------
    
    @requires_input_proccess
    def clone(self):
        toReturn = FullyConnectedModelSchema(input=self.input)
        toReturn.proccess_input_internal()
        toReturn.model.load_state_dict(self.model.state_dict())
        return toReturn
    
    

    # STATE MANAGEMENT -----------------------------------------------------
    
    def save_state(self):
        
        super().save_state()
        
        torch.save(self.model.state_dict(), os.path.join(self.artifact_relative_directory, "model_weights.pth"))
    
    
    
    def load_state(self) -> None:
        
        super().load_state()
                
        model_path = os.path.join(self.get_artifact_directory(), "model_weights.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at {model_path}")
                
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        self.initialize_model()  # Ensure the model is initialized before loading weights
        
        self.model.load_state_dict(state_dict) #loads the saved weights into the model
                