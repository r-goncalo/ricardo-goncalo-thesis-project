from automl.component import Schema, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

from abc import abstractmethod


class EnvironmentComponent(Schema):
    
    parameters_signature =  {} 
    
        
    @abstractmethod
    @requires_input_proccess    
    def reset(self):
        pass
        
    @abstractmethod
    @requires_input_proccess   
    def observe(self, *args):
        pass
        
    @abstractmethod
    @requires_input_proccess   
    def agents(self):
        pass
    
    
    @abstractmethod
    @requires_input_proccess  
    def action_space(self, *args):
        pass
    
    
    @abstractmethod
    @requires_input_proccess   
    def last(self):
        pass
        
    
    @abstractmethod
    @requires_input_proccess   
    def agent_iter(self):
        pass
    
    @requires_input_proccess    
    def step(self, *args):
        pass
        
    @requires_input_proccess    
    def rewards(self):
        pass    
    
    @requires_input_proccess
    def render(self):
        pass
    
    @requires_input_proccess
    def close(self):
        pass
     
    
from pettingzoo.butterfly import cooperative_pong_v5    

class PettingZooEnvironmentLoader(EnvironmentComponent):
    

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       "petting_zoo_environment" : InputSignature(default_value="cooperative_pong"),
                       "render_mode" : InputSignature(default_value="none", validity_verificator= lambda x : x in ["none", "human"]),
                       "device" : InputSignature(ignore_at_serialization=True)
                       }    
    
    def state_translator(state, device):
        
        with torch.no_grad():
            #return torch.from_numpy(state).to(torch.float32).to(device)
            return torch.tensor(state, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0

    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.device = self.input["device"]
        self.setup_environment()
        self.env.reset()
        
    def setup_environment(self):
        
        if self.input["petting_zoo_environment"] == "cooperative_pong":
        
            self.env = cooperative_pong_v5.env(render_mode=self.input["render_mode"])
            
        else:
            raise Exception(f"{self.name}: No valid petting zoo environment specified")
    
    def reset(self):
        
        super().reset()
        
        return self.env.reset()
        
    def observe(self, *args):
        super().reset()
        return PettingZooEnvironmentLoader.state_translator(self.env.observe(*args), self.device)
        
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
        return PettingZooEnvironmentLoader.state_translator(observation, self.device), reward, termination, info
    
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
     