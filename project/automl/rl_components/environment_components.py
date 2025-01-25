from ..component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

from abc import abstractmethod


class EnvironmentComponent(Component):
    
    input_signature =  {} 
     
    
from pettingzoo.butterfly import cooperative_pong_v5    

class PettingZooEnvironmentLoader(EnvironmentComponent):
    

    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = { "petting_zoo_environment" : InputSignature(default_value="cooperative_pong"),
                       "device" : InputSignature(ignore_at_serialization=True)
                       }    
    
    def state_translator(state, device):
        return torch.from_numpy(state).to(torch.float32).to(device)

    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.device = self.input["device"]
        self.setup_environment()
        self.env.reset()
        
    def setup_environment(self):
        
        if self.input["petting_zoo_environment"] == "cooperative_pong":
        
            self.env = cooperative_pong_v5.env(render_mode='none')
            
        else:
            raise Exception(f"{self.name}: No valid petting zoo environment specified")
    
    @requires_input_proccess    
    def reset(self):
        return self.env.reset()
        
    @requires_input_proccess    
    def observe(self, *args):
        return PettingZooEnvironmentLoader.state_translator(self.env.observe(*args), self.device)
        
    @requires_input_proccess    
    def agents(self):
        return self.env.agents
    
    @requires_input_proccess    
    def action_space(self, *args):
        return self.env.action_space(*args)
    
    @requires_input_proccess    
    def last(self):
        
        observation, reward, termination, truncation, info = self.env.last()
        
        #returns state, reward, done, info
        return PettingZooEnvironmentLoader.state_translator(observation, self.device), reward, termination, info
    
    @requires_input_proccess    
    def agent_iter(self):
        
        return self.env.agent_iter()
    
    @requires_input_proccess    
    def step(self, *args):
        
        return self.env.step(*args)
    
    @requires_input_proccess    
    def rewards(self):
        return self.env.rewards    