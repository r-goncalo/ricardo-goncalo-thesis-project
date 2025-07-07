import os
import traceback
from automl.basic_components.evaluator_component import ComponentWithEvaluator
from automl.basic_components.exec_component import ExecComponent
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_component_creation import get_sub_class_with_correct_parameter_signature
from automl.core.advanced_input_management import ComponentDictInputSignature, ComponentInputSignature, ComponentListInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.ml.memory.memory_components import MemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import EnvironmentComponent
from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper
from automl.utils.files_utils import open_or_create_folder
from automl.basic_components.state_management import StatefulComponent

import torch

import gc

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automl.utils.random_utils import generate_seed, do_full_setup_of_seed

class RLRePlayer(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent, ComponentWithEvaluator):
    
    '''A class which replays the actions in memory'''
    
    parameters_signature = {
                                                                                       
                       "environment" :  ComponentInputSignature(default_component_definition=(PettingZooEnvironmentWrapper, {})),
                       
                       "memory_list" : ComponentDictInputSignature(mandatory=True)
                       
                       }
    
    
    results_columns = ["episodes_done"] # this means a result_logger will exist with the column "episodes_done"

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super().proccess_input_internal()
        
        self.setup_environment()
        self.setup_memories_of_agents()


    def setup_environment(self):
        self.env : EnvironmentComponent = ComponentInputSignature.get_component_from_input(self, "environment")

    def setup_memories_of_agents(self):        
        self.memory_dict : dict[MemoryComponent] = ComponentDictInputSignature.get_component_list_from_input(self, "memory_list")



    def play_episode():
        pass

    def algorithm(self):
        
        super().algorithm()
        