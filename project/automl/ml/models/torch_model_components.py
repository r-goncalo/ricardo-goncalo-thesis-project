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
                


    
    
def perturb_model_parameters(
    torch_model: TorchModelComponent,
    min_percentage: float,
    max_percentage: float
):
    """
    Randomly perturbs model parameters within a given percentage range.

    @param min_percentage: float, e.g., 0.05 means perturbations will be at least ±5%.
    @param max_percentage: float, e.g., 0.1 means perturbations will be at most ±10%.
                           Each parameter will be multiplied by a random factor in
                           [0.9, 0.95] ∪ [1.05, 1.1] if min=0.05 and max=0.1.
    """

    if min_percentage < 0 or max_percentage < 0:
        raise ValueError("Percentages must be non-negative")
    if min_percentage > max_percentage:
        raise ValueError("min_percentage cannot be greater than max_percentage")

    if max_percentage == 0:
        return  # nothing to do

    if "perturbed_percentage" in torch_model.values.keys():
        print("WARNING: model has already had its parameters perturbed")

    torch_model.proccess_input_if_not_proccesd()

    torch_model.values["perturbed_percentage"] = (min_percentage, max_percentage)

    with torch.no_grad():
        for param in torch_model.model.parameters():
            # Generate random multipliers that enforce min_percentage
            random_sign = torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
            random_magnitude = torch.empty_like(param).uniform_(min_percentage, max_percentage)
            factor = 1.0 + random_sign * random_magnitude
            param.mul_(factor)



def perturb_model_parameters_gaussian(
    torch_model: TorchModelComponent,
    mean: float = 0.0,
    std: float = 0.1,
    fraction: float = 1.0
):
    """
    Perturbs model parameters by adding Gaussian noise.

    @param mean: Mean of the Gaussian noise (default = 0.0).
    @param std: Standard deviation of the Gaussian noise (default = 0.1).
    @param fraction: Fraction of parameters to perturb (default = 1.0, i.e. all).
                     For example, 0.1 perturbs 10% of parameters.
    """

    if std < 0:
        raise ValueError("Standard deviation must be non-negative")
    if not (0 < fraction <= 1.0):
        raise ValueError("Fraction must be in (0,1]")

    if "perturbed_gaussian" in torch_model.values.keys():
        print("WARNING: model has already had Gaussian noise added")

    torch_model.proccess_input_if_not_proccesd()
    torch_model.values["perturbed_gaussian"] = (mean, std, fraction)

    with torch.no_grad():
        for param in torch_model.model.parameters():
            if torch.rand(1).item() < fraction:
                noise = torch.randn_like(param) * std + mean
                param.add_(noise)