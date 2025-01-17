

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
from .optimizer_components import OptimizerComponent, AdamOptimizer

from .model_components import ConvModelComponent


# ACTUAL AGENT COMPONENT ---------------------------

from ..component import Component, InputSignature, requires_input_proccess
import torch
import random
import math
import numpy as nn

DEFAULT_MEMORY_SIZE = 200

class AgentComponent(Component):


    # INITIALIZATION --------------------------------------------------------------------------

    input_signature = { "name" : InputSignature(),
                       "device" : InputSignature(),
                       "logger" : InputSignature(),
                       "batch_size" : InputSignature(default_value=64),
                       "discount_factor" : InputSignature(default_value=0.95),
                       "target_update_rate" : InputSignature(default_value=0.05),
                       "training_context" : InputSignature(),
                       
                       "exploration_strategy" : InputSignature( generator=lambda _ : EpsilonGreedyStrategy()), #this generates an epsilon greddy strategy object at runtime if it is not specified
                       
                       "policy_model" : InputSignature(default_value=''),
                       "model_input_shape" : InputSignature(default_value='', description='The shape received by the model, only used when the model was not passed already initialized'),
                       "model_output_shape" : InputSignature(default_value='', description='Shape of the output of the model, only used when the model was not passed already'),
                       
                        
                       "optimizer" : InputSignature(generator= lambda x : AdamOptimizer()),
                       "replay_memory_size" : InputSignature(default_value=DEFAULT_MEMORY_SIZE)}

        
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
        self.lg = self.input["logger"]
        self.lg_profile = self.lg.createProfile(self.name)
    
        self.initialize_learning_strategy()
        self.initialize_models()
        self.initialize_optimizer()
        self.initialize_memory()    
            

    def initialize_learning_strategy(self):
                
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.GAMMA = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
        
        self.TAU = self.input["target_update_rate"] #the update rate of the target network
                
        self.lg_profile.writeLine(f"Batch size: {self.BATCH_SIZE} Gamma: {self.GAMMA} Tau: {self.TAU}")
        
        
        self.exploration_strategy = self.input["exploration_strategy"]
        
        self.exploration_strategy.pass_input(input= {"training_context" : self.input["training_context"]}) #the exploration strategy has access to the same training context
        
        
    def initialize_models(self):
        
        self.lg_profile.writeLine("Initializing policy model...")

        passed_policy_model = self.input["policy_model"]
        
        if passed_policy_model == '':
            passed_policy_model = self.create_policy_model()
            

        #our policy network will transform input frames to output acions (what should we do given the current frame?)
        self.policy_model : ConvModelComponent = passed_policy_model
        self.policy_model.pass_input({"device" : self.device}) #we have to guarantee that the device in our model is the same as the agent's

        self.lg_profile.writeLine("Initializing target model...")

        #our target network will be used to evaluate states
        #it is essentially a delayed copy of the policy network
        #it substitutes a Q table (a table that would store, for each state and each action, a value regarding how good that action is)
        self.target_net = self.policy_model.clone()


    #creates a policy model, only meant to be called if no policy model was passed
    def create_policy_model(self):        
                
        model_input_shape = self.input["model_input_shape"]
        model_output_shape = self.input["model_output_shape"]
        
        if model_input_shape != '' and model_output_shape != '':
            
            self.lg_profile.writeLine("Creating policy model using default values and passed shape...")
            
            #this makes some strong assumptions about the shape of the model and the input being received
            return ConvModelComponent(input={"board_x" : model_input_shape[0], "board_y" : model_input_shape[1], "board_z" : model_input_shape[2], "output_size" : model_output_shape})
            
            
        else:
            
            raise Exception('Undefined policy model for agent and undefined model input shape and output shape, used to create a default model')            
                            
                
    def initialize_optimizer(self):
        self.optimizer : OptimizerComponent = self.input["optimizer"]
        self.optimizer.pass_input({"model_params" : self.policy_model.get_model_params()})   
      
    
    def initialize_memory(self):
        
            
        replayMemorySize = self.input["replay_memory_size"]
        
        self.lg_profile.writeLine("Instantiating an empty memory with size " + str(replayMemorySize))
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
        
        self.lg_profile.saveFile(self.policy_model, 'model', 'policy_net')
        self.lg_profile.saveFile(self.target_net, 'model', 'target_net') 
        


