from automl.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
from automl.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion
from automl.loggers.global_logger import globalWriteLine
from automl.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn

from automl.component import InputSignature

from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space


class FullyConnectedModelSchema(TorchModelComponent):
    
    
    '''
        Represents a fully connected neural network model schema.
        
        The class "Model_Class" is the actual model architecture, which is a subclass of nn.Module
        A class that extends this schema could reimplement the Model_Class to define the architecture of the model.
    '''
        
    # The actual model architecture
    class Model_Class(nn.Module):
        
        def __init__(self, hidden_layers : list[int], activation_function="relu"):
            super().__init__()
            
            self.input_size = hidden_layers[0]
            self.output_size = hidden_layers[-1]


            if activation_function == "relu":
                activation_function  : type = nn.ReLU
            else:
                raise Exception(f"Unkown activation function")
            

            layers = []

            # for each layer except the last two, connect them with activation function
            for i in range(len(hidden_layers) - 2):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(activation_function())
            
            # the last is just a linear, without activation function
            layers.append(nn.Linear(hidden_layers[-2], hidden_layers[-1]))
            
            self.network = nn.Sequential(*layers)

        def forward(self, x : torch.Tensor):

            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.input_size) #the x is reshaped so it has 2 dimensions, the first one is the batch and the second the input size 
            
            return self.network(x)
        
        def get_output_shape(self):
            return self.output_size
        
        def get_input_shape(self):
            return self.input_size
    
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
                                    ),
        "activation_function" : InputSignature(default_value="relu")
    }    
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        

    def _setup_values(self):
        super()._setup_values()    

        self.activation_function = self.get_input_value("activation_function")

        self._setup_layers()
        

    def _setup_layers(self):

        self.hidden_size: int = self.get_input_value("hidden_size")
        self.hidden_layers: int = self.get_input_value("hidden_layers")
        self.layers = [*self.get_input_value("layers")]

        self.lg.writeLine(f"Model specification: hidden size: {self.hidden_size}, hidden_layers: {self.hidden_layers}, layers: {self.layers}")

        if (self.hidden_size is None or self.hidden_layers is None) and self.hidden_layers != self.hidden_size:
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

        elif self.layers is None and (self.hidden_size is None or self.hidden_layers is None):
            raise Exception(f"Must specify either hidden_layers and hidden_size or layers")

        if self.layers is None:
            self.layers = [self.hidden_size for _ in range(self.hidden_layers)]

        self._final_layers = [*self.layers]

        if self.input_shape is not None:
            input_size: int =  discrete_input_layer_size_of_space(self.input_shape)
            self._final_layers.insert(0, input_size)
            self.lg.writeLine(f"Input shape was specified: {self.input_shape}")
            self.lg.writeLine(f"First layer will have size of: {input_size}")


        if self.output_shape is not None:
            output_size = discrete_output_layer_size_of_space(self.output_shape)
            self._final_layers.append(output_size)
            self.lg.writeLine(f"Output shape was specified: {self.output_shape}")
            self.lg.writeLine(f"Last layer will have size of: {output_size}")
         
        self.lg.writeLine(f"Setup of layes of FCN over, layers are: {self._final_layers}")


    def _initialize_mininum_model_architecture(self):
    
        '''
        Initializes the model with no regard for initial parameters, as they are meant to be loaded
        This method is meant to be called even if the input isn't fully processed
        '''

        super()._initialize_mininum_model_architecture()

        self._setup_values() # this needs the values from the input fully setup

        self.model : nn.Module = type(self).Model_Class(
                hidden_layers=self._final_layers,
                activation_function=self.activation_function
            )

    def _initialize_model(self):

        '''Initializes the model with initial parameter strategy'''

        super()._initialize_model()

        self.model : nn.Module = type(self).Model_Class(
                hidden_layers=self._final_layers,
                activation_function=self.activation_function
            )
        
    def _is_model_well_formed(self):
        super()._is_model_well_formed()
        
        # TODO: verify if size and so on are coherent


    def _input_to_clone(self):
        input_to_clone = super()._input_to_clone()

        #input_to_clone.pop("hidden_layers", None)
        #input_to_clone.pop("hidden_size", None)
#
        #input_to_clone["layers"] = self.layers

        return input_to_clone
    


        
                            
    # EXPOSED METHODS --------------------------------------------