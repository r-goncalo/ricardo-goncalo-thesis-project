

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
                       "training_context" : InputSignature(
                           validity_verificator= lambda ctx : all(key in ctx.values.keys() for key in ["total_steps", "episodes_done"]))} #training context is a dictionary where we'll be able to get outside data   
    
    exposed_values = {"n_random" : 0, "n_greedy" : 0}
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.EPS_DECAY = self.input["epsilon_decay"]
        self.training_context = self.input["training_context"].values
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------
    
    def select_action(self, agent,  state):
                
        super().select_action(agent, state)
                
        sample = random.random()
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.training_context["episodes_done"] / self.EPS_DECAY)
        
        #in the case we use our policy net to predict our next action    
        if sample > eps_threshold:
            
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict(state)
        
        #in the case we choose a random action
        else:
            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict() 
        
        



from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.component import InputSignature

import random
import math
import torch

class EpsilonGreedyLinearStrategy(ExplorationStrategySchema):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { "epsilon_end" : InputSignature(default_value=0.025),
                            "exploration_fraction" : InputSignature(default_value=0.2),
                       "epsilon_start" : InputSignature(default_value=1.0),
                       "training_context" : InputSignature(
                           validity_verificator= lambda ctx : all(key in ctx.values.keys() for key in ["total_steps", "episodes_done"]))} #training context is a dictionary where we'll be able to get outside data   
    
    exposed_values = {"n_random" : 0, "n_greedy" : 0}
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.DECAY = self.input["exploration_fraction"]
        self.training_context = self.input["training_context"].values
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------
    
    def select_action(self, agent,  state):
                
        super().select_action(agent, state)
                
        sample = random.random()
        
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * max(0, 1 - self.DECAY * self.training_context["episodes_done"])
                
        #in the case we use our policy net to predict our next action    
        if sample > eps_threshold:
            
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict(state)
        
        #in the case we choose a random action
        else:
            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict() 
        