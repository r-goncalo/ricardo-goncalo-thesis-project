from ..component import Component
from ..component import input_signature
import torch
import random
import math
import numpy as nn

from abc import abstractmethod

class ModelComponent(Component):
    
    @abstractmethod
    def predict(self, state):
        
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

    input_signature = {**Component.input_signature, 
                       "board_x" : input_signature(),
                       "board_y" : input_signature(),
                       "board_z" : input_signature(),
                       "output_size" : input_signature()}    
    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.board_x = self.input["board_x"]                
        self.board_y = self.input["board_y"]
        self.board_z = self.input["board_z"]
        self.output_size = self.input["output_size"]
        
        self.model = ConvModelComponent.DQN(boardX=self.board_x, boardY=self.board_y, boardZ=self.board_z, n_actions=self.output_size)
        
    
    # UTIL -----------------------------------------------------
    
    def clone(self):
        
        toReturn =  ConvModelComponent(input=self.input)
        
        toReturn.model.load_state_dict(self.model.state_dict()) #copies current values into new model
        
            
    

        
    
    