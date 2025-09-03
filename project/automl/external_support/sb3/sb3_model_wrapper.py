
import copy
from automl.component import Component, InputSignature, requires_input_proccess

from automl.external_support.sb3.sb3_utils import load_sb3_dqn_model, load_sb3_q_net
from automl.ml.models.model_components import ModelComponent
from automl.ml.models.torch_model_components import TorchModelComponent
import torch

class SB3WrapperTorch(TorchModelComponent):
    
    parameters_signature = {
        "sb3_model" : InputSignature(default_value="dqn-MountainCar-v0", mandatory=False)
    }    
    
    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.sb3_model = self.input["sb3_model"]
    
    def _load_model(self):
        super()._load_model()
        # TODO: nn model from sb3 model if needed
    
    def _initialize_model(self):
        if not "sb3_model" in self.input.keys():
            Exception("No sb3_model_provided")
            
        
            
        # Clone q_net
        #sb3_model = load_sb3_dqn_model(self.input["sb3_model"])
        #q_net = sb3_model.policy.q_net
        q_net = load_sb3_q_net(self.input["sb3_model"])
                
        self.model = copy.deepcopy(q_net)
     
        # Explicitly load weights (not strictly needed, deepcopy already does it)
        self.model.load_state_dict(q_net.state_dict())
        
            
    @requires_input_proccess
    def predict(self, state):
        
        if state.ndim == 1:
            state = state.unsqueeze(0)   # make it (1, obs_dim)

        
        return self.model(state)
