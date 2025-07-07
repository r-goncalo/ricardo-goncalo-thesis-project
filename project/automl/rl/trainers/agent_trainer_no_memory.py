




from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.environment_components import EnvironmentComponent

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
import torch
import time

class AgentTrainerNoPreviousMemory(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {}
    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
    
        

    # INITIALIZATION ---------------------------------------------
    
    def initialize_agent(self):
    
        self.agent : AgentSchema = ComponentInputSignature.get_component_from_input(self, "agent")
        self.agent.proccess_input_if_not_proccesd()
        
        

    def initialize_memory(self):
        
        super().initialize_memory()
        self.memory.clear()
        
    
    # TRAINING_PROCESS ---------------------
        
    
    @requires_input_proccess
    def setup_episode(self, env : EnvironmentComponent):
        
        super().setup_episode(env)
        
        self.memory.clear()


    def optimizeAgent(self):
        
        super().optimizeAgent()
        
        self.memory.clear()        
