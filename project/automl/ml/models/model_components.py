from ...component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

class ModelComponent(Component):
    
    parameters_signature = {
        "input_shape": InputSignature(),
        "output_shape": InputSignature(),
    }    
    
    def proccess_input(self):
        super().proccess_input()
        
        self.input_shape = self.input["input_shape"]
        self.output_shape = self.input["output_shape"]
    
    
    def predict(self, state):
        pass
    
    @requires_input_proccess
    def get_model_params(self):
        '''returns a list of model parameters'''
        pass
    
    @requires_input_proccess
    def random_prediction(self):
        pass
    
    
    @requires_input_proccess            
    def update_model_with_target(self, target_model, target_model_weight):
        pass
    
    # UTIL -----------------------------------------------------
    
    @requires_input_proccess
    def clone(self):
        pass