

from automl.component import  InputSignature, requires_input_proccess



from automl.external_support.sb3.sb3_utils import load_sb3_dqn_model
from automl.rl.policy.policy import PolicyInterface


class SB3Wrapper(PolicyInterface):
    
    parameters_signature = {
        "sb3_model" : InputSignature(default_value="dqn-MountainCar-v0")
    }    
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        self.sb3_model = self.get_input_value("sb3_model")
        
        if isinstance(self.sb3_model, str):
            self.sb3_model = load_sb3_dqn_model(self.sb3_model)
    
    
    @requires_input_proccess
    def predict(self, state):
    
        action, hidden_state = self.sb3_model.predict(state, deterministic=True) #deterministic = True uses policy, deterministic = False simulates training behavior
        return action
    
    def get_policy_shape(self):
        raise NotImplementedError()
    
    
        
    @requires_input_proccess
    def random_prediction(self, state):
        action, hidden_state =  self.sb3_model.predict(state, deterministic=False)
        return action
    
    

    