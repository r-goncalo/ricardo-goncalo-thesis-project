from automl.component import Schema, InputSignature, requires_input_proccess
import torch
import numpy as nn

from abc import abstractmethod




class ExplorationStrategySchema(Schema):
    
    parameters_signature =  {
        "training_context" : InputSignature(possible_types=[dict]) # TODO: This should be substituted for a Component reference
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
    