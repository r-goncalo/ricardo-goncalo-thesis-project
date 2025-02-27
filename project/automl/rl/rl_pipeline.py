from automl.component import InputSignature, Schema, requires_input_proccess
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import EnvironmentComponent, PettingZooEnvironmentLoader
from automl.loggers.logger_component import LoggerSchema
from automl.utils.files_utils import open_or_create_folder

import torch

import gc

# TODO this is missing the evaluation component on a RLPipeline
class RLPipelineComponent(LoggerSchema):
    
    parameters_signature = {
        
                        "device" : InputSignature(default_value="cuda", ignore_at_serialization=True),
                                                        
                       "num_episodes" : InputSignature(),
                       
                       "environment" : InputSignature(generator= lambda self : self.initialize_child_component(PettingZooEnvironmentLoader)),
                       
                       "state_memory_size" : InputSignature(),
                       "agents" : InputSignature(default_value={}),
                       "agents_input" : InputSignature(default_value={}),
                       
                       "limit_steps" : InputSignature(),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100),
                       
                       "rl_trainer" : InputSignature(generator= lambda self : self.initialize_child_component(RLTrainerComponent))
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env : EnvironmentComponent= self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]        
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        self.configure_device(self.device)

        self.env.pass_input({"device" : self.device})
        
        self.initialize_agents_components()
    
        self.setup_trainer()

        for agent in self.agents.values(): #connect agents to rl_trainer
            agent.pass_input({"training_context" : self.rl_trainer}) 
    
    
        
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
            "agents" : self.agents.copy()
        }        
        
        self.rl_trainer : RLTrainerComponent = self.input["rl_trainer"]
        
        self.rl_trainer.pass_input(rl_trainer_input)
            
            
    def initialize_agents_components(self):

        self.agents = self.input["agents"] #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name

        if self.agents  == {}:
            self.create_agents()
        
        for agent_name, agent in self.agents.items():
            self.initialize_agent_component(agent_name, agent)                                                
                                                                                    
            
    def initialize_agent_component(self, agent_name, agent : AgentSchema):
            
            self.setup_agent_state_action_shape(agent_name, agent)
                        
            agent_logger = self.lg.openChildLog(logName=agent.input["name"])
            
            agent.pass_input({"logger_object" : agent_logger })
            agent.pass_input(self.input["agents_input"])
            
            
    def setup_agent_state_action_shape(self, agent_name, agent : AgentSchema):       
         
            state = self.env.observe(agent_name)
            
            self.lg.writeLine(f"State for agent {agent.name} has shape: {state.shape}")
            
            agent.pass_input({ "state_memory_size" : self.state_memory_size})

            action_shape = self.env.action_space(agent_name)
            self.lg.writeLine(f"Action space of agent {agent} has shape: {action_shape}")

            agent.pass_input({"state_shape" : state.shape })
            agent.pass_input({"action_shape" : action_shape })
            
            
            
    def create_agents(self):

        self.lg.writeLine("No agents defined, creating them...")

        agents = {}

        agentId = 1        
        for agent in self.env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {} #the input of the agent, with base values defined by "agents_input"

            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name    

            agents[agent] = self.initialize_child_component(AgentSchema, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg.writeLine("Initialized agents")

        self.agents : dict[str, AgentSchema] = agents  
        self.input["agents"] = agents #this is done because we want to save these agents in the configuration  
        
        
    # TRAINING_PROCCESS ----------------------
        
    @requires_input_proccess
    def train(self):        
        
        gc.collect() #this forces the garbage collector to collect any abandoned objects
        torch.cuda.empty_cache() #this clears cache of cuda
        
        self.rl_trainer.run_episodes()
        
        
    # RESULTS --------------------------------------
    
    def plot_graphs(self):
        
        self.rl_trainer.plot_results_graph()
        
    def get_last_Results(self):
        
        return self.rl_trainer.get_last_results()