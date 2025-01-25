from ..component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

from abc import abstractmethod

class ModelComponent(Component):
    
    @abstractmethod
    def predict(self, state):
        pass
    
    @abstractmethod
    def random_prediction(self):
        pass
    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F    

class ConvModelComponent(ModelComponent):
    
    
    #The actual model architecture
    class DQN(nn.Module):
        
        def __init__(self, boardX, boardY, boardZ, n_actions):
        
            super(ConvModelComponent.DQN, self).__init__()
            self.conv1 = nn.Conv2d(boardZ, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(boardX * boardY * 64, 512)
            self.fc2 = nn.Linear(512, n_actions)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1) if x.ndim > 3 else x.view(-1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x    
    

    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = {"board_x" : InputSignature(),
                       "board_y" : InputSignature(),
                       "board_z" : InputSignature(),
                       "output_size" : InputSignature(),
                       "device" : InputSignature(default_value="", ignore_at_serialization=True)}    
    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.board_x : int = self.input["board_x"]                
        self.board_y : int = self.input["board_y"]
        self.board_z : int = self.input["board_z"]
        self.output_size : int = self.input["output_size"]
        
        self.model = ConvModelComponent.DQN(boardX=self.board_x, boardY=self.board_y, boardZ=self.board_z, n_actions=self.output_size)
        
        if self.input["device"] != "":
            self.model.to(self.input["device"])
            
        print("Initializing model with input" + str(self.input))
                
    # EXPOSED METHODS --------------------------------------------
    
    @requires_input_proccess
    def get_model_params(self):
        '''returns a list of model parameters'''
        return list(self.model.parameters()) #the reason we use list is because parameters() returns an iterator (that is exaustable when iterated, not intended behaviour)
    
    @requires_input_proccess
    def predict(self, state):
        super().predict(state)
        toReturn = self.model(state)
        return toReturn
    
    
    @requires_input_proccess
    def random_prediction(self):
        super().random_prediction()
        
        return random.randrange(self.output_size) 
    
    
    @requires_input_proccess            
    def update_model_with_target(self, target_model, target_model_weight):
        
        '''
        Updates the weights of this model with the weights of the target model
        
        @param target_model_weight is the relevance of the target model, 1 will mean a total copy, 0 will do nothing, 0.5 will be an average between the models
        '''
        
        with torch.no_grad():
        
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            this_model_state_dict = self.model.state_dict()
            target_model_state_dict = target_model.model.state_dict()

            #the two models have the same shape and do
            for key in target_model_state_dict:
                this_model_state_dict[key] = target_model_state_dict[key] * target_model_weight + this_model_state_dict[key] * ( 1 - target_model_weight)

            self.model.load_state_dict(this_model_state_dict)
    
    # UTIL -----------------------------------------------------
    
    @requires_input_proccess
    def clone(self):
        
        print("Cloning model")
        
        toReturn = ConvModelComponent(input=self.input)
        toReturn.proccess_input()
        toReturn.model.load_state_dict(self.model.state_dict()) #copies current values into new model
        
        return toReturn
            
    

        
    
    