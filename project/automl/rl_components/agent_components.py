

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

from .model_components import ConvModelComponent


# ACTUAL AGENT COMPONENT ---------------------------

from ..component import Component
from ..component import input_signature
import torch
import random
import math
import numpy as nn

DEFAULT_MEMORY_SIZE = 200

class AgentComponent(Component):


    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = {**Component.input_signature, 
                       "name" : input_signature(),
                       "device" : input_signature(),
                       "logger" : input_signature(),
                       "batch_size" : input_signature(default_value=64),
                       "discount_factor" : input_signature(default_value=0.95),
                       "target_update_rate" : input_signature(default_value=0.05),
                       "learning_rate" : input_signature(default_value=0.01),
                       "exploration_strategy" : input_signature(generator=EpsilonGreedyStrategy), #this generates an epsilon greddy strategy object at runtime if it is not specified
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
        
        
    def initialize_models(self):
        
        self.lg.writeLine("Initializing policy model...")

        #our policy network will transform input frames to output acions (what should we do given the current frame?)
        self.policy_model = self.input["policy_model"]

        self.lg.writeLine("Initializing target model...")

        #our target network will be used to evaluate states
        #it is essentially a delayed copy of the policy network
        #it substitutes a Q table (a table that would store, for each state and each action, a value regarding how good that action is)
        self.target_net = self.policy_model.clone()
        
    def initialize_optimizer(self):
          self.optimizer = self.input["optimizer"]   
      
    
    def initialize_memory(self):
        
            
        replayMemorySize = self.input["replay_memory_size"]
        
        self.lg.writeLine("Instantiating an empty memory with size " + str(replayMemorySize))
        self.memory = ReplayMemory(replayMemorySize) #where we'll save the transitions we did    

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def policy_predict(self, state):
        return self.policy_model.predict(state)
    
    
    #selects action using policy prediction
    def select_action(self, state, training : dict):
        return self.exploration_strategy.select_action(state, self, training) #uses the exploration strategy defined, with the state, the agent and training information, to choose an action
            
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
        predicted_actions_values = self.policy_net(state_batch.to(device=self.device, dtype=torch.float32))
        predicted_actions_values, predicted_actions = predicted_actions_values.max(1)
        
        #compute the q values our target net predicts for the next_state (perceived reward)
        #if there is no next_state, we can use 0
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states.to(device=self.device, dtype=torch.float32)).max(1).values
            
        # Compute the expected Q values (the current reward of this state and the perceived reward we would get in the future)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(self, predicted_actions_values, expected_state_action_values)
                
    def update_target_model(self):
        
        with torch.no_grad():
        
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()

            #the two models have the same shape and do
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * ( 1 - self.TAU)

            self.target_net.load_state_dict(target_net_state_dict)
        
    
    # UTIL ----------------------------------------------------------------------------------------
    
    def saveModels(self):
        
        self.lg.saveFile(self.policy_net, 'model', 'policy_net')
        self.lg.saveFile(self.target_net, 'model', 'target_net') 
        
        
    def __str__(self):
        
        return self.name 
    


