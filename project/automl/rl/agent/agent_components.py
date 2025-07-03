
# DEFAULT COMPONENTS -------------------------------------

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema

from automl.ml.memory.memory_components import MemoryComponent

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema

from automl.rl.environment.environment_components import EnvironmentComponent

from automl.rl.policy.policy import Policy

from automl.basic_components.state_management import StatefulComponent

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

# ACTUAL AGENT COMPONENT ---------------------------

from automl.component import Component, InputSignature, requires_input_proccess
import torch
from automl.utils.class_util import get_class_from

from automl.utils.shapes_util import torch_zeros_for_space


DEFAULT_MEMORY_SIZE = 200

class AgentSchema(ComponentWithLogging, StatefulComponent):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                        "name" : InputSignature(),
                       "device" : InputSignature(get_from_parent=True, ignore_at_serialization=True),
                       "batch_size" : InputSignature(default_value=32),
                       "discount_factor" : InputSignature(default_value=0.95),
                       "training_context" : InputSignature(possible_types=[Component]),
                       
                       "exploration_strategy" : InputSignature( generator=lambda self : self.create_exploration_strategy(), priority=100), #this generates an epsilon greddy strategy object at runtime if it is not specified
                       "exploration_strategy_input" : InputSignature(default_value={}),
                       "exploration_strategy_class" : InputSignature(mandatory=False),
                                              
                       "state_shape" : InputSignature(default_value='', mandatory=False, description='The shape received by the model, only used when the model was not passed already initialized'),
                       "action_shape" : InputSignature(default_value='', mandatory=False, description='Shape of the output of the model, only used when the model was not passed already'),
                        
                       "memory" : ComponentInputSignature(
                            default_component_definition=(TorchMemoryComponent, {"capacity" : DEFAULT_MEMORY_SIZE}),
                       ),
                       
                       "learner" : ComponentInputSignature(
                            default_component_definition=(DeepQLearnerSchema, {})
                           ),
                                              
                       "policy" : ComponentInputSignature(
                            priority=100, mandatory=False, description="The policy to use for the agent, if not defined it will be created using the policy_class and policy_input"
                       ),
                       
                    }

        
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
    
        self.initialize_exploration_strategy()
        self.initialize_policy()
        self.initialize_learner()
        self.initialize_memory()
        
            

    def initialize_exploration_strategy(self):
                
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.discount_factor = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                
        self.lg.writeLine(f"Batch size: {self.BATCH_SIZE} discount_factor: {self.discount_factor}")
        
        self.exploration_strategy : ExplorationStrategySchema = self.input["exploration_strategy"]
        
        self.exploration_strategy.pass_input(input= {"training_context" : self.input["training_context"]}) #the exploration strategy has access to the same training context
        
        
    def create_exploration_strategy(self):
        
        '''creates and returns an exploration strategy for this agent'''       
        
        if not "exploration_strategy_class" in self.input.keys():
            raise Exception("No exploration strategy passed to agent and no exploration_strategy_class defined to generate it")
        
        self.exploration_strategy_class = self.input["exploration_strategy_class"]
                
        exploration_strategy_class : type[ExplorationStrategySchema] = get_class_from(self.exploration_strategy_class)
        
        return self.initialize_child_component(exploration_strategy_class, self.input["exploration_strategy_input"])         
        
        
        
    def initialize_policy(self):
        
        self.lg.writeLine("Initializing policy...")
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
    
        
        self.policy : Policy = ComponentInputSignature.get_component_from_input(self, "policy")
        
        self.policy.pass_input({"state_shape" : self.model_input_shape,
                               "action_shape" : self.model_output_shape,})
        

    def initialize_learner(self):
        
        self.learner : LearnerSchema = ComponentInputSignature.get_component_from_input(self, "learner")
        self.learner.pass_input({"device" : self.device, "agent" : self})
      
    
    def initialize_memory(self):
        
        self.memory : MemoryComponent = ComponentInputSignature.get_component_from_input(self, "memory")
        
            
        self.memory.pass_input({"state_dim" : self.model_input_shape, 
                                    "action_dim" : self.model_output_shape, 
                                    "device" : self.device}
                                )

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy
    
    @requires_input_proccess
    def policy_predict(self, state):
        
        '''makes a prediction based on the new state for a new action, using the current memory'''
        
        return self.policy.predict(state)
                    
    
    @requires_input_proccess
    def policy_random_predict(self):
        return self.policy.random_prediction()
    
    
    
    @requires_input_proccess
    #selects action using policy prediction
    def select_action(self, state):
        
         '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''
  
         return self.exploration_strategy.select_action(self, state)  
    
    
    @requires_input_proccess
    def optimize_policy_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        #a batch of transitions [(state, action next_state, reward)] transposed to [ (all states), (all actions), (all next states), (all rewards) ]
        batch = self.memory.sample_transposed(self.BATCH_SIZE)
        
                
        self.learner.learn(batch, self.discount_factor)
        
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_proccess
    def observe_transiction_to(self, new_state, action, reward):
        
        '''Makes agent observe and remember a transiction from a state to another'''
        
        prev_state_memory = torch.cat([element for element in self.state_memory])
        
        self.observe_new_state(new_state)
        
        next_state_memory = torch.cat([element for element in self.state_memory])
                
        self.memory.push(prev_state_memory, action, next_state_memory, reward)
        
    
    @requires_input_proccess
    def observe_new_state(self, new_state):
        pass
    
    @requires_input_proccess
    def reset_agent_in_environment(self, initial_state): # resets anything the agent has saved regarding the environment
        pass
         
             
         
        
        

