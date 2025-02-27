from ...component import Schema, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

class ModelComponent(Schema):
    
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
    
    def random_prediction(self):
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
    

#from automl.utils.class_util import get_class_from_string
#
## TODO: this could be done with multiple inheritence -> ModelLoader(ModelComponent, Loader)
#class ModelLoader(ModelComponent):
#    
#    '''An abstraction to select a type of model'''
#    
#    parameters_signature = {
#        "model_class_name" : InputSignature(),
#        "model_input" : InputSignature()
#    }
#    
#    def proccess_input(self):
#        super().proccess_input()
#        
#        self.model_class_name = self.input["model_class_name"]
#        
#        model_class : type[ModelComponent] = get_class_from_string(self.model_class_name)
#        
#        self.model = model_class(input=self.input["model_input"])
#        
#    def pass_input(self, input):
#        self.model.pass_input(input)
#     
#    
#    @requires_input_proccess
#    def predict(self, *args, **kwargs):
#        return self.model.predict(*args, **kwargs)
#    
#    @requires_input_proccess
#    def random_prediction(self, *args, **kwargs):
#        return self.model.random_prediction(*args, **kwargs)
#    
#    @requires_input_proccess
#    def get_model_params(self, *args, **kwargs):
#        return self.model.get_model_params(*args, **kwargs)
#    
#    
#    @requires_input_proccess            
#    def update_model_with_target(self, *args, **kwargs):
#        return self.model.update_model_with_target(*args, **kwargs)
#    
#    # UTIL -----------------------------------------------------
#    
#    @requires_input_proccess
#    def clone(self, *args, **kwargs):
#        return self.model.clone(*args, **kwargs)
#
#        
#    
#    