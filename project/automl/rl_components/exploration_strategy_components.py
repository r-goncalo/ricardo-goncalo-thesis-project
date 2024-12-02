from ..component import Component
from ..component import input_signature
import torch
import random
import math
import numpy as nn

from abc import abstractmethod




class ExplorationStrategyComponent(Component):
    
    @abstractmethod
    def select_action(self, agent, state, training : dict):
        
        '''
            Selects an action based on the agent's state (using things like its policy) and this exploration strategy
            
            Args:
                state is the current state as readable by the agent
                training is a dict with all the info the agent may need (for example, totalSteps)
                
            Returns:
                The index of the action selected
        '''
        
        pass
    
    
    
    
    

class EpsilonGreedyStrategy(ExplorationStrategyComponent):
    

    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = {**Component.input_signature, 
                       "epsilon_end" : input_signature(default_value=0.025),
                       "epsilon_start" : input_signature(default_value=1.0),
                       "epsilon_decay" : input_signature(default_value=0.01)}    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.EPS_DECAY = self.input["epsilon_decay"]
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------
    
    def select_action(self, agent,  state, training):
        
        super().select_action(state, training)
        
        sample = random.random()
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * training.totalSteps / self.EPS_DECAY)
            
        #in the case we use our policy net to predict our next action    
        if sample > eps_threshold:
            
            with torch.no_grad(): #we do not need to track de gradients because we won't do back propagation
                
                valuesForActions = agent.policy_predict(torch.tensor(state).to(device=self.device, dtype=torch.float32))
                max_value, max_index = valuesForActions.max(dim=0) 
                return torch.tensor([max_index], device=self.device).item()
        
        #in the case we choose a random action
        else:
            return torch.tensor([random.randrange(self.values["output_size"])], device=self.device, dtype=torch.long).item() 
        
        
        
        
        
        
class UpperConfidenceBoundStrategy(ExplorationStrategyComponent):
    
    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = {**Component.input_signature, 
                       "n_action" : input_signature(),
                       "exploration_param" : input_signature(default_value=0.01)}    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.EPS_DECAY = self.input["epsilon_decay"]
        
        self.counts = torch.zeros(self.values["output_size"], device=self.device)
        
        self.exploration_param  = self.values["exploration_param"]
        
        
        
    def select_action(self, agent, state, training):
        
        super().select_action(state, training)
        
        with torch.no_grad():
            
            q_values = agent.policy_predict(state)
                    
            #if we still have not tried a certain action
            if 0 in self.counts:
                action = torch.argmin(self.counts)
            
            else:
            
                #the value we choose is based on the predicted and
                ucb_values = q_values + self.exploration_param * torch.sqrt(math.log(training.totalSteps) / (self.counts))        
                action = torch.argmax(ucb_values)
            
            self.counts[action] += 1
            
            return action.item()