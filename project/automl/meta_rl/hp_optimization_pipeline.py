from automl.component import InputSignature, Schema, requires_input_proccess
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.model_components import ConvModelSchema
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import EnvironmentComponent, PettingZooEnvironmentLoader
from automl.loggers.logger_component import LoggerSchema
from automl.utils.files_utils import open_or_create_folder

import torch


class HyperparameterOptimizationPipeline(LoggerSchema):
    
    parameters_signature = {
        
                        "base_component" : InputSignature(),
                        
                        "evaluator" : InputSignature(),
                        
                        "hyperparameters_range_list" : InputSignature(
                            
                            default_value= [
                                (["agent_1", "agent_2"], ["discount_factor"], [0.5, 0.99]),
                                ([[2, 3, 0]], ["learning_rate"], [0.0001, 0.1])
                            ]
                            
                        )
                                                    
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.rl_pipeline : RLTrainerComponent = self.input["base_component"]
        
        # (component_names, input_keys, range) self.hyperparameters_range_list = 
        
        self.hyperparameters_range_list : list[tuple[list[str], list[str], any]] = 
    
        
    def configure_device(self, str_device_str):
        
        try:

            self.lg.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg.writeLine("The model will trained and evaluated on: " + str(self.device))
        
    def setup_trainer(self):    
        
        rl_trainer_input = {
            "device" : self.device,
            "logger_object" : self.lg,
            "num_episodes" : self.num_episodes,
            "state_memory_size" : self.state_memory_size,
            "environment" : self.env,
            "limit_steps" : self.limit_steps ,
            "optimization_interval" : self.optimization_interval,
            "agents" : self.agents
        }        
        
        self.rl_trainer : RLTrainerComponent = self.input["rl_trainer"]
        
        self.rl_trainer.pass_input(rl_trainer_input)
            
            
    def initialize_agents_components(self):

        self.agents = self.input["agents"] #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name

        if self.agents  == {}:
            self.create_agents()
        
    def create_agents(self):

        self.lg.writeLine("Creating agents")

        agents = {}

        agentId = 1        
        for agent in self.env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {**self.input["created_agents_input"]} #the input of the agent, with base values defined by "agents_input"


            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name

            agent_logger = self.lg.openChildLog(logName=agent_name)
            agent_input["logger_object"] = agent_logger

            state = self.env.observe(agent)

            z_input_size = len(state)
            y_input_size = len(state[0])
            x_input_size = len(state[0][0])
            
            self.lg.writeLine("State for agent " + agent_name + " has shape: Z: " + str(z_input_size) + " Y: " + str(y_input_size) + " X: " + str(x_input_size))
            
            agent_input["state_memory_size"] = self.state_memory_size

            n_actions = self.env.action_space(agent).n
            print(f"Action space of agent {agent}: {self.env.action_space(agent)}")

            agent_input["state_shape"] = [x_input_size, y_input_size, z_input_size]
            agent_input["action_shape"] = n_actions
            
            agent_input["device"] = self.device       

            agents[agent] = self.initialize_child_component(AgentSchema, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg.writeLine("Initialized " + str(agents) + " agents")

        self.agents = agents  
        self.input["agents"] = agents #this is done because we want to save these agents in the configuration
        
        
        
    # TRAINING_PROCCESS ----------------------
        
    @requires_input_proccess
    def train(self):        
        self.rl_trainer.run_episodes()
        
        
    # RESULTS --------------------------------------
    
    def plot_graphs(self):
        
        self.rl_trainer.plot_results_graph()