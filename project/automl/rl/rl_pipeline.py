import os
import traceback
from automl.basic_components.evaluator_component import ComponentWithEvaluator
from automl.basic_components.exec_component import ExecComponent
from automl.basic_components.seeded_component import SeededComponent
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_component_creation import get_sub_class_with_correct_parameter_signature
from automl.loggers.component_with_results import ComponentWithResults
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

from automl.core.advanced_input_management import ComponentInputSignature

from automl.utils.random_utils import generate_seed, do_full_setup_of_seed

# TODO this is missing the evaluation component on a RLPipeline
class RLPipelineComponent(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent, ComponentWithEvaluator, SeededComponent):
    
    parameters_signature = {
        
                        "device" : InputSignature(default_value="cuda", ignore_at_serialization=True),
                                                                               
                       "environment" : ComponentInputSignature(default_component_definition=(PettingZooEnvironmentWrapper, {}), possible_types=[EnvironmentComponent]),
                       
                       "agents" : InputSignature(default_value={}),
                       "agents_input" : InputSignature(default_value={}),

                       "save_in_between" : InputSignature(default_value=True),
                       
                       "rl_trainer" : ComponentInputSignature(default_component_definition=(RLTrainerComponent, {})),
                
                       }
    
    
    results_columns = ["episodes_done"] # this means a result_logger will exist with the column "episodes_done"

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        print(f"Environment: {self.input['environment']}")
        
        super().proccess_input_internal()
                
        self.device = self.input["device"]
        
        self.configure_device(self.device)
        
        self.setup_environment()
                        
        self.save_in_between = self.input["save_in_between"]
        
        self.initialize_agents_components()
    
        self.setup_trainer()
        
        self.rl_setup_evaluator()
        
    def setup_environment(self):
        self.env : EnvironmentComponent = ComponentInputSignature.get_component_from_input(self, "environment")
                
        self.env.pass_input({"device" : self.device})
        
        self.env.proccess_input_if_not_proccesd()

    
    
    def rl_setup_evaluator(self):
        
        '''Setups custom logic for '''
        
        if self.component_evaluator is not None:
                
            evaluation_columns = self.component_evaluator.get_metrics_strings()
        
            self.add_to_columns_of_results_logger(evaluation_columns)
    
        
    def configure_device(self, str_device_str):
        
        '''Configures the torch device'''
        
        try:

            self.lg.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg.writeLine("The model will trained and evaluated on: " + str(self.device))
        
        
    def setup_trainer(self):    
        
        '''Setups the trainer, which is rensonsible for executing the training algorithm, agnostic to the learning methods, models and number of agents'''
                
        rl_trainer_input = {
            "device" : self.device,
            "logger_object" : self.lg,
            "environment" : self.env,
            "agents" : self.agents.copy()
        }        
        
        self.rl_trainer : RLTrainerComponent = ComponentInputSignature.get_component_from_input(self, "rl_trainer")
        
        self.rl_trainer.pass_input(rl_trainer_input)
            
            
    def initialize_agents_components(self):
        
        '''Initialize the agents, creating them if necessary first'''

        self.agents = self.input["agents"] #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name

        if self.agents  == {}:
            self.create_agents()
        
        for agent_name, agent in self.agents.items():
            self.configure_agent_component(agent_name, agent)                                                
                                                                                    
            
    def configure_agent_component(self, agent_name, agent : AgentSchema):
        
        '''Configures the agents, setting up their action and state spaces, the logger and more'''
            
        self.setup_agent_state_action_shape(agent_name, agent)
                            
        agent.pass_input(self.input["agents_input"])
            
            
    def setup_agent_state_action_shape(self, agent_name, agent : AgentSchema):       
        
        '''Setups the agent state space and action shape'''
         
        state_shape = self.env.get_agent_state_space(agent_name)
                
        self.lg.writeLine(f"State for agent {agent.name} has shape state: {state_shape}")

        action_shape = self.env.get_agent_action_space(agent_name)

        self.lg.writeLine(f"Action space of agent {agent.name} has shape: {action_shape}")

        agent.pass_input({"state_shape" : state_shape })
        agent.pass_input({"action_shape" : action_shape })
            
            
            
    def create_agents(self):
        
        '''Creates agents'''

        self.lg.writeLine("No agents defined, creating them...")

        agents = {}

        agentId = 1        
        for agent in self.env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {} #the input of the agent, with base values defined by "agents_input"

            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name
            
            agent_input["base_directory"] = os.path.join(self.get_artifact_directory(), "agents" )
            
            agent_class = get_sub_class_with_correct_parameter_signature(AgentSchema, self.input["agents_input"]) #gets the agent class with the correct parameter signature

            agents[agent] = self.initialize_child_component(agent_class, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg.writeLine("Initialized agents")

        self.agents : dict[str, AgentSchema] = agents  
        self.input["agents"] = agents #this is done because we want to save these agents in the configuration  
        
    
        
    # TRAINING_PROCCESS ----------------------
    
    def onException(self, exception : Exception):
        
        super().onException(exception)
        
        error_message = str(exception)
        full_traceback = traceback.format_exc()

        self.lg.writeLine("Error message:", file="error_report.txt")
        self.lg.writeLine(error_message, file="error_report.txt")

        self.lg.writeLine("\nFull traceback:")
        self.lg.writeLine(full_traceback, file="error_report.txt")
        
        
    @requires_input_proccess
    def train(self):        
        '''Executes the training part of the algorithm with a specified number of episodes < than the number specified'''
        
        gc.collect() #this forces the garbage collector to collect any abandoned objects
        torch.cuda.empty_cache() #this clears cache of cuda
        
        self.rl_trainer.run_episodes()
        
        if self.save_in_between:
            self.save_state()
        
    
    @requires_input_proccess
    def algorithm(self):
        
        '''
        Executes the training part of the algorithm with a specified number of episodes < than the number specified
        
        It then evaluates and returns the results
        '''
        
        self.train() #trains the agents in the reinforcement learning pipeline
        
        
        if self.component_evaluator is not None:
            
            self.lg.writeLine("Evaluating the trained agents...")
            
            evaluation_results = self.evaluate_this_component() # evaluates the resulting model and saves the results

            self.log_results({
            "episodes_done" : self.rl_trainer.values["episodes_done"],
                          **evaluation_results})
            
        else:
            self.log_results({
                "episodes_done" : self.rl_trainer.values["episodes_done"],
            })
        
        
    
        
    # RESULTS --------------------------------------
    
    @requires_input_proccess
    def get_results_logger(self):
        return self.rl_trainer.get_results_logger()
    
    
    
    #@requires_input_proccess    
    #def get_last_Results(self):
    #    
    #    return self.rl_trainer.get_last_results()