

from automl.component import Component, InputSignature, requires_input_proccess
from automl.rl.environment.environment_components import EnvironmentComponent

import torch
import random
import math
import numpy as nn


class PettingZooEnvironmentWrapper(EnvironmentComponent):
    
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       "environment" : InputSignature(default_value="cooperative_pong"),
                       "render_mode" : InputSignature(default_value="none", validity_verificator= lambda x : x in ["none", "human"]),
                       "device" : InputSignature(ignore_at_serialization=True)
                       }    
    
    def state_translator(state, device):
        
        with torch.no_grad():
            #return torch.from_numpy(state).to(torch.float32).to(device)
            return torch.tensor(state, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0

    
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.device = self.input["device"]
        self.setup_environment()
        self.env.reset()
        
    
    def setup_environment(self):
        
        from pettingzoo import ParallelEnv
        
        if isinstance(self.input["environment"], str):
            self.load_environment(self.input["environment"])
            
        elif isinstance(self.input["environment"], ParallelEnv):
            self.env = self.input["environment"]
            
        else:
            raise Exception("No valid environment or environment name passed to PettingZoo Wrapper")
        
        
    def load_environment(self, environment_name : str):
        
        if environment_name == "cooperative_pong":
            
            from pettingzoo.butterfly import cooperative_pong_v5
            self.env = cooperative_pong_v5.env(render_mode=self.input["render_mode"])
            
        else:
            raise Exception(f"{self.name}: No valid petting zoo environment specified")
        
        
    
    def reset(self):
        
        super().reset()
        
        return self.env.reset()
    
        
    def observe(self, *args):
        super().reset()
        return PettingZooEnvironmentWrapper.state_translator(self.env.observe(*args), self.device)
    
        
    def agents(self):
        super().reset()
        return self.env.agents
    
    
    def action_space(self, *args):
        super().reset()
        return self.env.action_space(*args)
    
    
    def last(self):
        super().reset()
        observation, reward, termination, truncation, info = self.env.last()
        
        #returns state, reward, done, info
        return PettingZooEnvironmentWrapper.state_translator(observation, self.device), reward, termination, info
    
    def agent_iter(self):
        super().reset()
        return self.env.agent_iter()
    
    def step(self, *args):
        super().reset()
        return self.env.step(*args)
    
    def rewards(self):
        super().reset()
        return self.env.rewards    

    def render(self):
        self.env.render()
    

    def close(self):
        self.env.close()