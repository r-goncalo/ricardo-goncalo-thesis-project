

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.component import InputSignature

import random
import math
import torch


class EpsilonGreedyStrategy(ExplorationStrategySchema):
    

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { "epsilon_end" : InputSignature(default_value=0.025),
                       "epsilon_start" : InputSignature(default_value=1.0),
                       "epsilon_decay" : InputSignature(default_value=0.01),
                       "training_context" : InputSignature(validity_verificator= lambda ctx : all(key in ctx.keys() for key in ["total_steps"]))} #training context is a dictionary where we'll be able to get outside data   
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.EPS_DECAY = self.input["epsilon_decay"]
        self.training_context = self.input["training_context"]
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------
    
    def select_action(self, agent,  state):
                
        super().select_action(agent, state)
                
        sample = random.random()
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.training_context["total_steps"] / self.EPS_DECAY)
        
        #in the case we use our policy net to predict our next action    
        if sample > eps_threshold:
            
            with torch.no_grad(): #we do not need to track de gradients because we won't do back propagation
                                
                valuesForActions = agent.policy_predict(state)
                max_value, max_index = valuesForActions.max(dim=0) 
                return max_index.item()
        
        #in the case we choose a random action
        else:
            return agent.policy_random_predict() 
        