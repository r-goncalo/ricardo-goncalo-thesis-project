
# DEFAULT COMPONENTS -------------------------------------

from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.learner_component import DeepQLearnerSchema

from automl.ml.models.model_components import ConvModelSchema

from automl.rl.memory_components import MemoryComponent

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema

from automl.rl.environment.environment_components import EnvironmentComponent

# ACTUAL AGENT COMPONENT ---------------------------

from automl.component import Schema, InputSignature, requires_input_proccess, uses_component_exception
from automl.loggers.logger_component import LoggerSchema
import torch
import random
import math
import numpy as nn

import wandb

DEFAULT_MEMORY_SIZE = 200

class AgentSchema(LoggerSchema):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { "name" : InputSignature(),
                       "device" : InputSignature(generator= lambda self : self.get_attr_from_parent("device"), ignore_at_serialization=True),
                       "batch_size" : InputSignature(default_value=64),
                       "discount_factor" : InputSignature(default_value=0.95),
                       "training_context" : InputSignature(),
                       
                       "exploration_strategy" : InputSignature( generator=lambda self : self.initialize_child_component(EpsilonGreedyStrategy)), #this generates an epsilon greddy strategy object at runtime if it is not specified
                       
                       "policy_model" : InputSignature(priority=50, generator= lambda self : self.create_policy_model()),
                       "state_shape" : InputSignature(default_value='', description='The shape received by the model, only used when the model was not passed already initialized'),
                       "action_shape" : InputSignature(default_value='', description='Shape of the output of the model, only used when the model was not passed already'),
                        
                       "memory" : InputSignature(generator = lambda self :  self.initialize_child_component(MemoryComponent, input={"capacity" : DEFAULT_MEMORY_SIZE})),
                       
                       "learner" : InputSignature(generator= lambda self : self.initialize_child_component(DeepQLearnerSchema)),
                       
                       "state_memory_size" : InputSignature(default_value=1, description="This makes the agent remember previous states of the environment and concatenates them")
                    }

        
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
    
        self.initialize_exploration_strategy()
        self.initialize_state_memory()
        self.initialize_models()
        self.initialize_learner()
        self.initialize_memory()
        
            

    def initialize_exploration_strategy(self):
                
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.GAMMA = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                
        self.lg.writeLine(f"Batch size: {self.BATCH_SIZE} Gamma: {self.GAMMA}")
        
        
        self.exploration_strategy : ExplorationStrategySchema = self.input["exploration_strategy"]
        
        self.exploration_strategy.pass_input(input= {"training_context" : self.input["training_context"]}) #the exploration strategy has access to the same training context
        
        
    def initialize_models(self):
        
        self.lg.writeLine("Initializing policy model...")

        passed_policy_model = self.input["policy_model"]
        
        #our policy network will transform input frames to output acions (what should we do given the current frame?)
        self.policy_model : ConvModelSchema = passed_policy_model
        self.policy_model.pass_input({"device" : self.device}) #we have to guarantee that the device in our model is the same as the agent's


    def initialize_state_memory(self):
        
        if hasattr(self, "state_memory_size"):
            pass #the state memory size was already initialized
        
        else:
        
            self.state_memory_size = self.input["state_memory_size"]
            self.state_memory_list = [None] * self.state_memory_size

            #if self.state_memory_size > 1:

            self.lg.writeLine(f"Initializing agent with more than one state memory size ({self.state_memory_size})")

            if self.input["state_shape"] == '':
                raise Exception("More than one state memory size and undefined model input shape")

            self.state_length = self.input["state_shape"][2]

            self.lg.writeLine(f"State length is {self.state_length}")


    #creates a policy model, only meant to be called if no policy model was passed
    def create_policy_model(self):        
                
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.initialize_state_memory() #we need to initialize the 
        
        if self.model_input_shape != '' and self.model_output_shape != '':
            
            model_input = {"board_x" : int(self.model_input_shape[0]), 
                           "board_y" : int(self.model_input_shape[1]), 
                           "board_z" : int(self.model_input_shape[2]) * self.state_memory_size, 
                           "output_size" : int(self.model_output_shape)}
            
            self.lg.writeLine("Creating policy model using default values and passed shape... Model input: " + str(model_input))
            
            #this makes some strong assumptions about the shape of the model and the input being received
            return self.initialize_child_component(ConvModelSchema, input=model_input)
            
        else:
            
            raise Exception('Undefined policy model for agent and undefined model input shape and output shape, used to create a default model')            

    def initialize_learner(self):
        
        self.learner : LearnerSchema = self.input["learner"]
        
        self.learner.pass_input({"device" : self.device, "agent" : self})
      
    
    def initialize_memory(self):
        
        self.memory : MemoryComponent = self.input["memory"] #where we'll save the transitions we did    

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy_model
    
    @requires_input_proccess
    @uses_component_exception
    def policy_predict(self, state):
        self.update_state_memory(state)
        
        if self.state_memory_size > 1:
        
            return self.policy_model.predict(torch.cat(self.state_memory_list))
        
        else:
            
            return self.policy_model.predict(self.state_memory_list)
    
    @requires_input_proccess
    @uses_component_exception
    def policy_random_predict(self):
        return self.policy_model.random_prediction()
    
    @requires_input_proccess
    @uses_component_exception
    #selects action using policy prediction
    def select_action(self, state):
        self.update_state_memory(state)
         #uses the exploration strategy defined, with the state, the agent and training information, to choose an action

        if self.state_memory_size > 1:
        
            return self.exploration_strategy.select_action(self, torch.cat(self.state_memory_list) )
        
        else:
            
            return self.exploration_strategy.select_action(self, self.state_memory_list )  
    
    
    @requires_input_proccess
    @uses_component_exception        
    def optimize_policy_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        #a batch of transitions [(state, action next_state, reward)]
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch
        #[ (all states), (all actions), (all next states), (all rewards) ]
        batch = self.memory.Transition(*zip(*transitions))
                
        self.learner.learn(batch, self.GAMMA)
        
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_proccess
    def observe_transiction_to(self, new_state, action, reward):
        
        prev_state_memory_list = [element for element in self.state_memory_list]
        prev_state_memory = torch.cat(prev_state_memory_list)
        
        self.update_state_memory(new_state)
        
        next_state_memory_list = [element for element in self.state_memory_list]
        next_state_memory = torch.cat(next_state_memory_list) 
        
        #print(f"prev state len: {len(prev_state_memory)} net state len: {len(next_state_memory)}")
        
        self.memory.push(prev_state_memory, action, next_state_memory, reward)
    
    @requires_input_proccess
    def observe_new_state(self, new_state):
        self.update_state_memory(new_state)
    
    @requires_input_proccess
    def reset_state_memory(self, new_state): #setup memory shared accross agents
        
        if self.state_memory_size > 1:
                    
            self.state_memory_list = [new_state] * self.state_memory_size
            
        else:
                        
            self.state_memory_list = new_state
         
             
    @requires_input_proccess    
    def update_state_memory(self, new_state): #update memory shared accross agents
                   
        if self.state_memory_size > 1:   
            
            for i in range(1, self.state_memory_size):
                self.state_memory_list[i - 1] = self.state_memory_list[i]
            
            self.state_memory_list[self.state_memory_size - 1] = new_state
        
        else:
            
            self.state_memory_list = new_state


    
    # UTIL ----------------------------------------------------------------------------------------
    
    @requires_input_proccess
    @uses_component_exception
    def save_policy(self):
        
        self.lg.saveFile(self.policy_model, 'model', 'policy_net')
        


