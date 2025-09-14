import torch
import torch.nn as nn

from automl.component import InputSignature, requires_input_proccess
from automl.ml.models.model_components import ModelComponent

class TorchModelComponent(ModelComponent):
    
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "device": InputSignature(get_from_parent=True, ignore_at_serialization=True),
        "model" : InputSignature(mandatory=False, possible_types=[nn.Module])
    }    
    
    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self._setup_values()
                
        self._setup_model()
        
        if "device" in self.input.keys():
            self.model.to(self.input["device"])
        
        
    def _setup_values(self):
        pass
        
    def _setup_model(self):
        
        model_loaded = self._is_model_loaded()

        if self._should_load_model():
            
            if model_loaded:
                print("WARNING: LOADING A TORCH MODEL WHEN A MODEL WAS ALREADY LOADED")
        
            self._load_model()
            
        else:
            self._initialize_model() # initializes the model using passed values

        self._is_model_well_formed() # throws exception if model is not well formed


    def _is_model_well_formed(self):
        if not self._is_model_loaded():
            raise Exception("Failed to load the model")
            
            
    def _is_model_loaded(self):
        return hasattr(self, "model") and self.model is not None
    
    def _should_load_model(self):
        return "model" in self.input.keys()
                

    def _load_model(self):
        
        if "model" in self.input.keys():
            self.model : nn.Module = self.input["model"]
        
    def _initialize_model(self):
        raise Exception("Model initialization not implemented in base TorchModelComponent")

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
        toReturn = type(self)(input=self.input) # initializes the component
        toReturn.proccess_input_internal()
        toReturn.model.load_state_dict(self.model.state_dict())
        return toReturn
                


    
    
def perturb_model_parameters(torch_model : TorchModelComponent, percentage: float):
    """
    Randomly perturbs model parameters within a given percentage range.
    @param percentage: float, e.g., 0.1 for 10%, means each parameter will be multiplied
                       by a random factor in [0.9, 1.1].
    """
    
    if percentage < 0:
        raise ValueError("Percentage must be positive or 0")
    
    elif percentage == 0:
        pass
    
    else: # if percentage > 0

        if "perturbed_percentage" in torch_model.values.keys():
            print("WARNING: model has already had its parameters perturbed")
            
        torch_model.proccess_input_if_not_proccesd()
    
        torch_model.values["perturbed_percentage"] = percentage # TODO: this should be multiplicative
    
        with torch.no_grad():
            for param in torch_model.model.parameters():
                # Generate random multiplier in [1 - percentage, 1 + percentage]
                factor = torch.empty_like(param).uniform_(1 - percentage, 1 + percentage)
                param.mul_(factor)