from automl.component import Component, InputSignature, requires_input_proccess
import torch
import numpy as nn

from abc import abstractmethod




class ExplorationStrategySchema(Component):
    
    parameters_signature =  {
        "training_context" : InputSignature(possible_types=[Component])
        } 

    
    @abstractmethod
    @requires_input_proccess
    def select_action(self, agent, state):
        
        '''
            Selects an action based on the agent's state (using things like its policy) and this exploration strategy
            
            Args:
                state is the current state as readable by the agent
                
            Returns:
                The index of the action selected
        '''
        
        pass
    