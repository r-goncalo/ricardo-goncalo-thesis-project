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

    exposed_values = {
        "model" : None
    }
    
    def proccess_input_internal(self):
        
        super().proccess_input_internal()

        self.__synchro_model_value_attr()
        
        self._setup_values()
                
        self._setup_model()
        
        if "device" in self.input.keys():
            self.model.to(self.input["device"])

    def __synchro_model_value_attr(self):
        if self.values["model"] is not None: # if a model is already present in the exposed values, use it
            self.model = self.values["model"]
        
        elif self.values["model"] is None and hasattr(self, "model"): # if a model is already present as an attribute, use it
            self.values["model"] = self.model
        
        
    def _setup_values(self):
        pass
        
    def _setup_model(self):
        
        model_loaded = self._is_model_loaded()

        if self._should_load_model():
            
            if model_loaded:
                print("WARNING: LOADING A TORCH MODEL WHEN A MODEL WAS ALREADY LOADED, WILL NOT LOAD NEW MODEL AND KEEP THE OLD ONE")

            else:
                self._load_model()

        else:
            self._initialize_model() # initializes the model using passed values

        self.__synchro_model_value_attr()

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
            self.values["model"] = self.model
        
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
        toReturn.proccess_input_if_not_proccesd()
        toReturn.model.load_state_dict(self.model.state_dict())
        return toReturn
                