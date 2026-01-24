

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.component import InputSignature

import random
import math
import torch

from automl.rl.agent.agent_components import AgentSchema

class EpsilonGreedyStrategy(ExplorationStrategySchema):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { "epsilon_end" : InputSignature(default_value=0.025,
                                custom_dict={
                                    "hyperparameter_suggestion" : ("float", {"low" : 0.001, "high" : 0.15})
                                }),
                       "epsilon_start" : InputSignature(default_value=1.0),
                       "epsilon_decay" : InputSignature(default_value=0.01),
                       "training_context" : InputSignature(
                           validity_verificator= lambda ctx : all(key in ctx.values.keys() for key in ["total_steps", "episodes_done"]))} #training context is a dictionary where we'll be able to get outside data   
    
    exposed_values = {"n_random" : 0, "n_greedy" : 0}
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
        self.EPS_END = self.get_input_value("epsilon_end")                
        self.EPS_START = self.get_input_value("epsilon_start")
        self.EPS_DECAY = self.get_input_value("epsilon_decay")
        self.training_context = self.get_input_value("training_context").values
    

    
    # EXPOSED METHOD --------------------------------------------------------------------------

    def should_select_random_action(self):

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.training_context["episodes_done"] / self.EPS_DECAY)

        sample = random.random()

        return sample > eps_threshold

    
    def select_action(self, agent : AgentSchema,  state):
                
        super().select_action(agent, state)
        
        #in the case we use our policy net to predict our next action    
        if self.should_select_random_action():
            
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict(state)
        
        #in the case we choose a random action
        else:
            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict() 
        
    
    def select_action_with_memory(self, agent : AgentSchema):

        super().select_action_with_memory(agent)
        
        #in the case we use our policy net to predict our next action    
        if self.should_select_random_action():
            
            self.values["n_greedy"] = self.values["n_greedy"] + 1
            return  agent.policy_predict_with_memory()
        
        #in the case we choose a random action
        else:
            self.values["n_random"] = self.values["n_random"] + 1
            return agent.policy_random_predict()         
        


class EpsilonGreedyLinearStrategy(EpsilonGreedyStrategy):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { }
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
    
    # EXPOSED METHOD --------------------------------------------------------------------------

    def should_select_random_action(self):

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * max(0, 1 - self.EPS_DECAY * self.training_context["episodes_done"])

        sample = random.random()

        return sample > eps_threshold
    
        