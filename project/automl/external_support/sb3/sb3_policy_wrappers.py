

from automl.component import  InputSignature, requires_input_proccess



from automl.rl.policy.policy import PolicyInterface


class SB3Wrapper(PolicyInterface):
    
    parameters_signature = {
        "sb3_model" : InputSignature(default_value="dqn-MountainCar-v0")
    }    
    
    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.sb3_model = self.input["sb3_model"]
        
        if isinstance(self.sb3_model, str):
            self.sb3_model = self.__load_sb3_model(self.sb3_model)
    
    
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
    
    
    def __load_sb3_model(self, model_name : str):
        
        from huggingface_sb3 import load_from_hub

        checkpoint = load_from_hub(
        	repo_id=f"sb3/{model_name}",
        	filename=f"{model_name}.zip",
        )

        from stable_baselines3 import DQN

        model_sb3 = DQN.load(checkpoint)
        
        return model_sb3
    
    