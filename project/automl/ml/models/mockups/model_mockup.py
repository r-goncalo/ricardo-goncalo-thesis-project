

from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.ml.models.model_components import ModelComponent
from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space
import torch


class MockupRandomModel(ModelComponent):
    
    parameters_signature = {
        "input_shape": InputSignature(),
        "output_shape": InputSignature(),
        "device": InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }    
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.input_size = discrete_input_layer_size_of_space(self.input_shape)
        self.output_size: int = discrete_output_layer_size_of_space(self.output_shape)
        self.device = self.input["device"]
        
        
    
    @requires_input_proccess
    def predict(self, state):
        
        total_elements = state.numel()
        
        if total_elements % self.input_size != 0:
            
            raise ValueError(f"Incompatible input shape {state.shape} for input size {self.input_size}")
        batch_size = total_elements // self.input_size
        
        return torch.empty(batch_size, self.output_size, device=self.device)
    
    @requires_input_proccess
    def get_model_params(self):
        '''returns a list of model parameters'''
        return []
    
    
    @requires_input_proccess            
    def update_model_with_target(self, target_model, target_model_weight):
        pass
    
    # UTIL -----------------------------------------------------
    
    @requires_input_proccess
    def clone(self):
        return self