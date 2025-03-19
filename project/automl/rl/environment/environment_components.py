from automl.component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn


class EnvironmentComponent(Component):
    
    parameters_signature =  {} 
    
        
    @requires_input_proccess    
    def reset(self):
        pass
        
    @requires_input_proccess   
    def observe(self, *args):
        pass
        
    @requires_input_proccess   
    def agents(self):
        pass
    
    
    @requires_input_proccess  
    def action_space(self, *args):
        pass
    
    @requires_input_proccess   
    def last(self):
        pass
    
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
     
    

     