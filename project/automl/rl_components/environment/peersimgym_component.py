from ...component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn
from .environment_components import EnvironmentComponent

from abc import abstractmethod


# TODO: Implement this


#import peersim_gym.envs.PeersimEnv import PeersimEnv


class PeersimGymComponent(EnvironmentComponent):
    
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
        
        self.env = PeersimEnv()
        
    
    def reset(self):
        return self.env.reset()
        
    def observe(self, *args):
        return PeersimGymComponent.state_translator(self.env.observe(*args), self.device)
        
    def agents(self):
        return self.env.agents
    
    def action_space(self, *args):
        return self.env.action_space(*args)
    
    def last(self):
        
        observation, reward, termination, truncation, info = self.env.last()
        
        #returns state, reward, done, info
        return PeersimGymComponent.state_translator(observation, self.device), reward, termination, info
    
    def agent_iter(self):
        
        return self.env.agent_iter()
    
    def step(self, *args):
        
        return self.env.step(*args)
    
    def rewards(self):
        return self.env.rewards    