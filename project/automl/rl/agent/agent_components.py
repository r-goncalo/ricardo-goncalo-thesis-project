
# DEFAULT COMPONENTS -------------------------------------

from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import DeepQLearnerSchema

from automl.ml.models.model_components import ConvModelSchema

from automl.rl.memory_components import MemoryComponent

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
                       
                       "policy_model" : InputSignature(priority=2, generator= lambda self : self.create_policy_model()),
                       "model_input_shape" : InputSignature(default_value='', description='The shape received by the model, only used when the model was not passed already initialized'),
                       "model_output_shape" : InputSignature(default_value='', description='Shape of the output of the model, only used when the model was not passed already'),
                        
                       "memory" : InputSignature(generator = lambda self :  self.initialize_child_component(MemoryComponent, input={"capacity" : DEFAULT_MEMORY_SIZE})),
                       
                       "learner" : InputSignature(generator= lambda self : self.initialize_child_component(DeepQLearnerSchema))
                    }

        
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
    
        self.initialize_exploration_strategy()
        self.initialize_models()
        self.initialize_learner()
        self.initialize_memory()
            

    def initialize_exploration_strategy(self):
                
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.GAMMA = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                
        self.lg.writeLine(f"Batch size: {self.BATCH_SIZE} Gamma: {self.GAMMA}")
        
        
        self.exploration_strategy = self.input["exploration_strategy"]
        
        self.exploration_strategy.pass_input(input= {"training_context" : self.input["training_context"]}) #the exploration strategy has access to the same training context
        
        
    def initialize_models(self):
        
        self.lg.writeLine("Initializing policy model...")

        passed_policy_model = self.input["policy_model"]
        
        #our policy network will transform input frames to output acions (what should we do given the current frame?)
        self.policy_model : ConvModelSchema = passed_policy_model
        self.policy_model.pass_input({"device" : self.device}) #we have to guarantee that the device in our model is the same as the agent's


    #creates a policy model, only meant to be called if no policy model was passed
    def create_policy_model(self):        
                
        model_input_shape = self.input["model_input_shape"]
        model_output_shape = self.input["model_output_shape"]
        
        if model_input_shape != '' and model_output_shape != '':
            
            print(type(model_input_shape[0]))
            print(type(model_output_shape))
            
            model_input = {"board_x" : int(model_input_shape[0]), "board_y" : int(model_input_shape[1]), "board_z" : int(model_input_shape[2]), "output_size" : int(model_output_shape)}
            
            self.lg.writeLine("Creating policy model using default values and passed shape... Model input: " + str(model_input))
            
            #this makes some strong assumptions about the shape of the model and the input being received
            return self.initialize_child_component(ConvModelSchema, input=model_input)
            
        else:
            
            raise Exception('Undefined policy model for agent and undefined model input shape and output shape, used to create a default model')            

    def initialize_learner(self):
        
        self.learner = self.input["learner"]
        
        self.learner.pass_input({"device" : self.device, "agent" : self})
      
    
    def initialize_memory(self):
        
        self.memory = self.input["memory"] #where we'll save the transitions we did    

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy_model
    
    @requires_input_proccess
    @uses_component_exception
    def policy_predict(self, state):
        return self.policy_model.predict(state)
    
    @requires_input_proccess
    @uses_component_exception
    def policy_random_predict(self):
        return self.policy_model.random_prediction()
    
    @requires_input_proccess
    @uses_component_exception
    #selects action using policy prediction
    def select_action(self, state):
        return self.exploration_strategy.select_action(self, state) #uses the exploration strategy defined, with the state, the agent and training information, to choose an action
    

    def do_training_step(self, i_episode, env, state_memory_size, state):

        action = self.select_action(state) # decides the next action to take (can be random)
                                     
        env.step(action) #makes the game proccess the action that was taken
        
        boardObs, reward, done, info = self.env.last()
        
        total_score += reward
                                        
        if done:
            next_state = None
        else:
            
            if self.state_memory_size > 1: #if we have memory in our states (we use previous states as inputs for actions)
            
                next_state = torch.stack([  state[i][u] for u in range(0, state_memory_size)  for i in range(1, state_memory_size)]) #adds the previous perceived states to the memory of the next state
                for window in boardObs:
                    next_state = torch.stack((next_state, window)) #adds the new perceived state
            else:
                
                next_state = boardObs #if we have no memory (if we just use the current state)
                                    
    
        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)
        
        # Save the (next) previous state
        state = next_state
        
        if self.values["total_steps"] % self.optimization_interval == 0:
            
            self.lg.writeLine(f"In episode {i_episode}, optimizing at step {t} that is the total step {self.values['total_steps']}")
            self.optimizeAgents()
            
        t += 1
        self.values["total_steps"] += 1 #we just did a step


    
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
        

    
    # UTIL ----------------------------------------------------------------------------------------
    
    @requires_input_proccess
    @uses_component_exception
    def save_policy(self):
        
        self.lg.saveFile(self.policy_model, 'model', 'policy_net')
        


