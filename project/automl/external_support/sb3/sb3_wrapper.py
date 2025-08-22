
from automl.component import Component, InputSignature, requires_input_proccess

from automl.ml.models.model_components import ModelComponent

class SB3Wrapper(ModelComponent):
    
    parameters_signature = {
        "sb3_model" : InputSignature()
    }    
    
    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.sb3_model = self.input["sb3_model"]
    
    
    def predict(self, state):
        return self.sb3_model.predict(state, deterministic=True) #deterministic = True uses policy, deterministic = False simulates training behavior
    
    @requires_input_proccess
    def get_model_params(self):
        raise Exception("Sb3 model can't offer its model parameters")