




from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.aec_environment import AECEnvironmentComponent

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.utils.shapes_util import discrete_output_layer_size_of_space
import torch
import time

class AgentTrainerDQN(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                        "exploration_strategy" : ComponentInputSignature(mandatory=False
                                                        ),
                       
                       }        
        

    # INITIALIZATION ---------------------------------------------
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.initialize_exploration_strategy()
        

    def initialize_exploration_strategy(self):
                
        if "exploration_strategy" in self.input.keys():
            
            self.exploration_strategy : ExplorationStrategySchema = self.get_input_value("exploration_strategy") 
            self.exploration_strategy.pass_input(input= {"training_context" : self}) #the exploration strategy has access to the same training context
        
            self.lg.writeLine(f"Using exploration strategy {self.exploration_strategy.name}")
            
        else:
            self.lg.writeLine(f"No exploration strategy defined")
            
            self.exploration_strategy = None

        

    def initialize_memory(self):
        
        super().initialize_memory()
        
        
        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("state", self.agent.model_input_shape), 
                                        ("action", self.agent.get_policy().get_policy_shape(), torch.int64),
                                        ("next_state", self.agent.model_input_shape),
                                        ("reward", 1),
                                        ("done", 1)
                                    ]
            
        self.memory.pass_input({
                                    "transition_data" : self.memory_fields_shapes
                                })
        

    @requires_input_proccess
    def end_training(self):        
        super().end_training()
        
        if self.exploration_strategy is not None:
            self.lg.writeLine(f"Exploration strategy values: \n{self.exploration_strategy.values}\n")
        

    def _observe_transiction_to(self, new_state, action, reward, done):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''
        
        
        self.state_memory_temp.copy_(self.agent.get_current_state_in_memory())
        
        self.agent.update_state_memory(new_state)
        
        next_state_memory = self.agent.get_current_state_in_memory()
                
        self.memory.push({"state" : self.state_memory_temp, "action" : action, "next_state" : next_state_memory, "reward" : reward, "done" : done})
        

        

    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''

        if self.exploration_strategy is not None:
  
            return self.exploration_strategy.select_action(self.agent, state)  

        else:
            return super().select_action(self.agent, state)