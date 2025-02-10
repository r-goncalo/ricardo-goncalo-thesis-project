from ..component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

from abc import abstractmethod




class ExplorationStrategyComponent(Component):
    
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
    
    
    
    
    

class EpsilonGreedyStrategy(ExplorationStrategyComponent):
    

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
        
        
        
        
        
        
class UpperConfidenceBoundStrategy(ExplorationStrategyComponent):
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {"n_action" : InputSignature(),
                       "exploration_param" : InputSignature(default_value=0.01)}    
    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.EPS_END = self.input["epsilon_end"]                
        self.EPS_START = self.input["epsilon_start"]
        self.EPS_DECAY = self.input["epsilon_decay"]
        
        self.counts = torch.zeros(self.values["output_size"], device=self.device)
        
        self.exploration_param  = self.values["exploration_param"]
        
        
        
    def select_action(self, agent, state):
        
        super().select_action(state)
        
        with torch.no_grad():
            
            q_values = agent.policy_predict(state)
                    
            #if we still have not tried a certain action
            if 0 in self.counts:
                action = torch.argmin(self.counts)
            
            else:
            
                #the value we choose is based on the predicted and
                ucb_values = q_values + self.exploration_param * torch.sqrt(math.log(self.training_context["total_steps"]) / (self.counts))        
                action = torch.argmax(ucb_values)
            
            self.counts[action] += 1
            
            return action.item()