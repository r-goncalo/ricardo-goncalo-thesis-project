from automl.component import Component, InputSignature, requires_input_proccess

from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer

from automl.utils.shapes_util import discrete_input_layer_size_of_space, discrete_output_layer_size_of_space

from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from

import torch

import random

class Policy(Component):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''
        
    parameters_signature = {
        "model" : InputSignature(mandatory=False),
        "model_class" : InputSignature(mandatory=False),
        "model_input" : InputSignature(default_value={}),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature(),
    }   
    
    def proccess_input(self):
        
        super().proccess_input()
        self.initialize_model()
        
        
    def initialize_model(self):
        
        if not "model" in self.input.keys():
            self.model : ModelComponent = self.create_model()
        
        else:
            self.model : ModelComponent = self.input["model"]

            
        self.model.pass_input(self.input['model_input'])
        
        
        
    def create_model(self):
        
        if not "model_class" in self.input.keys():
            raise Exception("Model not defined and model class not defined")
        
        model_class = get_class_from(self.input["model_class"])
        
        return self.initialize_child_component(model_class)
        
        
    def predict(self, state):
        '''Uses the state and the policy's model to predict an action'''
        pass
    
    
    def random_prediction(self, state):
        pass
    

class QPolicy(Policy):
    '''
    It uses a model to calculate the Q action values of each action and calculates the action as such
    '''
        
    parameters_signature = {
        "model" : InputSignature(mandatory=False),
        "model_class" : InputSignature(mandatory=False),
        "model_input" : InputSignature(default_value={}),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature()
    }   
    
    def proccess_input(self):
        
        super().proccess_input()
        self.initialize_model()
        
        
    def initialize_model(self):
    
        super().initialize_model()
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.model.pass_input({"input_shape" : self.model_input_shape, "output_shape" : self.model_output_shape})
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    @requires_input_proccess
    def predict(self, state):
        
        valuesForActions : torch.Tensor = self.model.predict(state)
        
        max_value, max_index = valuesForActions.max(dim=1)
        
        return max_index.item() # the action
    
    
    @requires_input_proccess
    def random_prediction(self):
        return random.randint(0, self.model_output_shape.n - 1)