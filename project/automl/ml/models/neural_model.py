import torch
import torch.nn as nn
import torch.nn.functional as F    

from automl.component import Schema, InputSignature, requires_input_proccess
import random

from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space

from automl.ml.models.model_components import ModelComponent

class FullyConnectedModelSchema(ModelComponent):
        
    # The actual model architecture
    class DQN(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, hidden_layers):
            super(FullyConnectedModelSchema.DQN, self).__init__()
            
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
                #x = x.view(x.size(0), -1)  # Ensures batch processing compatibility
                #x = x.view(-1)
                x = x.view(-1, self.input_size)
                        
            return self.network(x)
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "hidden_layers" : InputSignature(description="Number of hidden layers"),
        "hidden_size": InputSignature(description="Size of hidden layers"),
        "device": InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }    
    
    def proccess_input(self):
        
        super().proccess_input()
        
        self.input_size: int =  discrete_input_layer_size_of_space(self.input_shape)
        
        self.hidden_size: int = self.input["hidden_size"]
        self.hidden_layers: int = self.input["hidden_layers"]
        
        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)
                
        self.model = FullyConnectedModelSchema.DQN(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size,
            hidden_layers=self.hidden_layers
        )
        
        if "device" in self.input.keys():
            self.model.to(self.input["device"])
            
        print("Initializing model with input" + str(self.input))
                
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
    def random_prediction(self):
        super().random_prediction()
        return random.randrange(self.output_size) 
    
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
        toReturn.proccess_input()
        toReturn.model.load_state_dict(self.model.state_dict())
        return toReturn
