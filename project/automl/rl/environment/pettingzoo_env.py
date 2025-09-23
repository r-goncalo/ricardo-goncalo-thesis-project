

from automl.basic_components.seeded_component import SeededComponent
from automl.component import Component, InputSignature, requires_input_proccess
from automl.rl.environment.environment_components import EnvironmentComponent

from automl.rl.environment.gymnasium_env import GymnasiumEnvironmentWrapper
import torch

from pettingzoo import ParallelEnv


# TODO: This should probably extend Gymnasium
class PettingZooEnvironmentWrapper(GymnasiumEnvironmentWrapper):
        
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       "environment" : InputSignature(default_value="cooperative_pong"),
                       "render_mode" : InputSignature(default_value="none", validity_verificator= lambda x : x in ["none", "human"]),
                       "device" : InputSignature(ignore_at_serialization=True)
                       }    
    
    
    @staticmethod
    def state_translator(state, device):
        
        with torch.no_grad():
            #return torch.from_numpy(state).to(torch.float32).to(device)
            return torch.tensor(state, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0

    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
        self.device = self.input["device"]
        self.setup_environment()
        self.env.reset()
        
    
    def setup_environment(self):
                
        if isinstance(self.input["environment"], str):
            self.load_environment(self.input["environment"])
            
        elif isinstance(self.input["environment"], ParallelEnv):
            self.env : ParallelEnv = self.input["environment"]
            
        else:
            raise Exception("No valid environment or environment name passed to PettingZoo Wrapper")
        
        
    def load_environment(self, environment_name : str):
        
        if environment_name == "cooperative_pong":
            
            from pettingzoo.butterfly import cooperative_pong_v5
            self.env : ParallelEnv = cooperative_pong_v5.env(render_mode=self.input["render_mode"])
            
        else:
            raise Exception(f"{self.name}: No valid petting zoo environment specified")
        
    
        
    def observe(self, *args):
        return PettingZooEnvironmentWrapper.state_translator(self.env.observe(*args), self.device)
    
    
    @requires_input_proccess
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        return self.env.action_space(agent)
    
    @requires_input_proccess
    def get_agent_state_space(self, agent):
        '''returns the state space for the environment'''
        raise NotImplementedError()
    
    
    @requires_input_proccess
    def agents(self):
        return self.env.agents
    
    
    def last(self):
        observation, reward, termination, truncation, info = self.env.last()
        
        #returns state, reward, done, info
        return PettingZooEnvironmentWrapper.state_translator(observation, self.device), reward, truncation or termination, info
    
    def agent_iter(self):
        return self.env.agent_iter()
    
    def step(self, *args):
        return self.env.step(*args)
    
    def rewards(self):
        return self.env.rewards    

    def render(self):
        self.env.render()
    

    def close(self):
        self.env.close()
        
    def reset(self):
        observations, info = self.env.reset()
        self.reset_info = info
        return observations
