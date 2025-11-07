


from automl.ml.models.torch_model_components import TorchModelComponent
import torch
import torch.nn as nn
import torch.nn.functional as F    

from ...component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

from automl.ml.models.model_components import ModelComponent

class ConvModelSchema(TorchModelComponent):
    
    
    #The actual model architecture
    class DQN(nn.Module):
        
        def __init__(self, boardX, boardY, boardZ, n_actions):
        
            super(ConvModelSchema.DQN, self).__init__()
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

    parameters_signature = {
        "board_x" : InputSignature(),
                       "board_y" : InputSignature(),
                       "board_z" : InputSignature(),
                       "output_size" : InputSignature(),
                       "device" : InputSignature(default_value="", ignore_at_serialization=True)}    
    
    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
        self.board_x : int = self.get_input_value("board_x")                
        self.board_y : int = self.get_input_value("board_y")
        self.board_z : int = self.get_input_value("board_z")
        self.output_size : int = self.get_input_value("output_size")
        
        self.model = ConvModelSchema.DQN(boardX=self.board_x, boardY=self.board_y, boardZ=self.board_z, n_actions=self.output_size)
        

    
    