from automl.component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn


class EnvironmentComponent(Component):
    
    parameters_signature =  {} 
    
        
    @requires_input_proccess    
    def reset(self):
        raise NotImplementedError()
        
    @requires_input_proccess   
    def observe(self, *args):
        raise NotImplementedError()
        
    @requires_input_proccess   
    def agents(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_agent_state_space(self, agent):
        '''returns the state space for the environment'''
        raise NotImplementedError()
    
    @requires_input_proccess   
    def last(self):
        raise NotImplementedError()
    
    @requires_input_proccess   
    def agent_iter(self):
        raise NotImplementedError()
    
    @requires_input_proccess    
    def step(self, *args):
        raise NotImplementedError()
        
    @requires_input_proccess    
    def rewards(self):
        raise NotImplementedError()    
    
    @requires_input_proccess
    def render(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def close(self):
        raise NotImplementedError()
    
     
    

     