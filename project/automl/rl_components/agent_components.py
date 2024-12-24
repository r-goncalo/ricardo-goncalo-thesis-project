

# UTIL --------------------------------

from collections import namedtuple
from collections import deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    #save a transition
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# DEFAULT COMPONENTS -------------------------------------

from .exploration_strategy_components import EpsilonGreedyStrategy
from .optimizer_components import OptimizerComponent

from .model_components import ConvModelComponent


# ACTUAL AGENT COMPONENT ---------------------------

from ..component import Component, input_signature, requires_input_proccess
import torch
import random
import math
import numpy as nn

DEFAULT_MEMORY_SIZE = 200

class AgentComponent(Component):


    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = { "name" : input_signature(),
                       "device" : input_signature(),
                       "logger" : input_signature(),
                       "batch_size" : input_signature(default_value=64),
                       "discount_factor" : input_signature(default_value=0.95),
                       "target_update_rate" : input_signature(default_value=0.05),
                       "learning_rate" : input_signature(default_value=0.01),
                       "training_context" : input_signature(),
                       
                       "exploration_strategy" : input_signature( generator=lambda _ : EpsilonGreedyStrategy()), #this generates an epsilon greddy strategy object at runtime if it is not specified
                       
                       "policy_model" : input_signature(), 
                       "optimizer" : input_signature(),
                       "replay_memory_size" : input_signature(default_value=DEFAULT_MEMORY_SIZE)}

        
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
        self.lg = self.input["logger"]
    
        self.initialize_learning_strategy()
        self.initialize_models()
        self.initialize_optimizer()
        self.initialize_memory()    
            

    def initialize_learning_strategy(self):
                
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.GAMMA = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
        
        self.TAU = self.input["target_update_rate"] #the update rate of the target network
        
        self.LR = self.input["learning_rate"] #the learning rate of the optimizer  
        
        self.lg.writeLine(f"Batch size: {self.BATCH_SIZE} Gamma: {self.GAMMA} Tau: {self.TAU} Learning rate: {self.LR}")
        
        
        self.exploration_strategy = self.input["exploration_strategy"]
        
        self.exploration_strategy.pass_input(input= {"training_context" : self.input["training_context"]}) #the exploration strategy has access to the same training context
        
        
    def initialize_models(self):
        
        self.lg.writeLine("Initializing policy model...")

        #our policy network will transform input frames to output acions (what should we do given the current frame?)
        self.policy_model : ConvModelComponent = self.input["policy_model"]
        self.policy_model.pass_input({"device" : self.device}) #we have to guarantee that the device in our model is the same as the agent's

        self.lg.writeLine("Initializing target model...")

        #our target network will be used to evaluate states
        #it is essentially a delayed copy of the policy network
        #it substitutes a Q table (a table that would store, for each state and each action, a value regarding how good that action is)
        self.target_net = self.policy_model.clone()
        
                
    def initialize_optimizer(self):
        self.optimizer : OptimizerComponent = self.input["optimizer"]
        self.optimizer.pass_input({"model_params" : self.policy_model.get_model_params()})   
      
    
    def initialize_memory(self):
        
            
        replayMemorySize = self.input["replay_memory_size"]
        
        self.lg.writeLine("Instantiating an empty memory with size " + str(replayMemorySize))
        self.memory = ReplayMemory(replayMemorySize) #where we'll save the transitions we did    

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    @requires_input_proccess
    def policy_predict(self, state):
        return self.policy_model.predict(state)
    
    @requires_input_proccess
    def policy_random_predict(self):
        return self.policy_model.random_prediction()
    
    @requires_input_proccess
    #selects action using policy prediction
    def select_action(self, state):
        return self.exploration_strategy.select_action(self, state) #uses the exploration strategy defined, with the state, the agent and training information, to choose an action
    
    @requires_input_proccess        
    def optimize_policy_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        #a batch of transitions [(state, action next_state, reward)]
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch
        #[ (all states), (all actions), (all next states), (all rewards) ]
        batch = Transition(*zip(*transitions))
                
        state_batch = torch.stack(batch.state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                        if s is not None], dim=0)
                
        
        #predict the action we would take given the current state
        predicted_actions_values = self.policy_model.predict(state_batch)
        predicted_actions_values, predicted_actions = predicted_actions_values.max(1)
        
        #compute the q values our target net predicts for the next_state (perceived reward)
        #if there is no next_state, we can use 0
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.predict(non_final_next_states).max(1).values
            
        # Compute the expected Q values (the current reward of this state and the perceived reward we would get in the future)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        print("Gradients of predicted: " + str(predicted_actions_values.grad_fn))
        
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_actions_values, expected_state_action_values)
             
    @requires_input_proccess            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.policy_model, self.TAU)
        
    
    # UTIL ----------------------------------------------------------------------------------------
    
    @requires_input_proccess
    def saveModels(self):
        
        self.lg.saveFile(self.policy_model, 'model', 'policy_net')
        self.lg.saveFile(self.target_net, 'model', 'target_net') 
        


