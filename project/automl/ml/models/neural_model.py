import os
from automl.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
from automl.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion
from automl.loggers.global_logger import globalWriteLine
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
        
        def __init__(self, input_size, output_size, hidden_layers : list[int]):
            super(FullyConnectedModelSchema.Model_Class, self).__init__()
            
            self.input_size = input_size
            
            layers = []
            prev_size = input_size
            
            for i in range(len(hidden_layers)):
                layers.append(nn.Linear(prev_size, hidden_layers[i]))
                layers.append(nn.ReLU())
                prev_size = hidden_layers[i]
            
            layers.append(nn.Linear(prev_size, output_size))
            
            self.network = nn.Sequential(*layers)

        def forward(self, x : torch.Tensor):

            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.input_size) #the x is reshaped so it has 2 dimensions, the first one is the batch and the second the input size 
                        
            return self.network(x)
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "hidden_layers" : InputSignature(mandatory=False, description="Number of hidden layers"),
        "hidden_size": InputSignature(mandatory=False, description="Size of hidden layers"),
        "layers" : InputSignature(mandatory=False, 
                                  custom_dict={"hyperparameter_suggestion" : 
                                               VariableListHyperparameterSuggestion(
                                                   name="layers",
                                                   min_len=2,
                                                   max_len=4,
                                                   hyperparameter_suggestion_for_list=
                                                   SingleHyperparameterSuggestion(
                                                       value_suggestion=("cat", {"choices" : [16, 32, 64, 128, 256]})
                                                   ))
                                            }
                                    )
    }    
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        

    def _setup_values(self):
        super()._setup_values()    

        if self.input_shape == None:
            raise Exception(f"{type(self)} needs input shape to be passed to setup its values, input: {self.input}")
        
        if self.output_shape == None:
            raise Exception(f"{type(self)} needs output shape to be passed to setup its values, input: {self.input}")
        
        self.input_size: int =  discrete_input_layer_size_of_space(self.input_shape)
        
        self._setup_layers()
        
        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)

    def _setup_layers(self):

        self.hidden_size: int = self.get_input_value("hidden_size")
        self.hidden_layers: int = self.get_input_value("hidden_layers")
        self.layers = self.get_input_value("layers")

        if self.hidden_size is None or self.hidden_layers is None and self.hidden_layers != self.hidden_size:
            self.lg.writeLine(f"{self.name}: had hidden layers {self.hidden_layers} and hidden size {self.hidden_size}, both should not be None to be used")
            self.remove_input("hidden_layers")
            self.remove_input("hidden_size")
            self.hidden_size = None
            self.hidden_layers = None
        
        elif self.hidden_size is None and self.hidden_layers is None and self.layers is not None:
            self.lg.writeLine(f"{self.name}: had hidden layers {self.hidden_layers} and hidden size {self.hidden_size}, but layers were defined, using layers...")
            self.remove_input("hidden_layers")
            self.remove_input("hidden_size")
            self.hidden_size = None
            self.hidden_layers = None

        elif self.layers is None and self.hidden_size is None and self.hidden_layers is None:
            raise Exception(f"Must specify either hidden_layers and hidden_size or layers")


        if self.layers is None:
            self.layers = [self.hidden_size for _ in range(self.hidden_layers)]

        self.lg.writeLine(f"Setup of layes of FCN over, layers are: {self.layers}")

        


    def _initialize_mininum_model_architecture(self):
    
        '''
        Initializes the model with no regard for initial parameters, as they are meant to be loaded
        This method is meant to be called even if the input isn't fully processed
        '''

        super()._initialize_mininum_model_architecture()

        self._setup_values() # this needs the values from the input fully setup

        self.model : nn.Module = type(self).Model_Class(
            input_size=self.input_size, 
                output_size=self.output_size,
                hidden_layers=self.layers
            )

    def _initialize_model(self):

        '''Initializes the model with initial parameter strategy'''

        super()._initialize_model()

        self.model : nn.Module = type(self).Model_Class(
            input_size=self.input_size,
                output_size=self.output_size,
                hidden_layers=self.layers
            )
        
    def _is_model_well_formed(self):
        super()._is_model_well_formed()
        
        # TODO: verify if size and so on are coherent
        
                            
    # EXPOSED METHODS --------------------------------------------