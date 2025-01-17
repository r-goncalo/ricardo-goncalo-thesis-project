from ..component import InputSignature, Component, requires_input_proccess
from .agent_components import AgentComponent
from .optimizer_components import AdamOptimizer
from .exploration_strategy_components import EpsilonGreedyStrategy
from .model_components import ConvModelComponent
from .rl_trainer_component import RLTrainerComponent

import torch
import time

class RLPipelineComponent(Component):

    TRAIN_LOG = 'train.txt'
    
    input_signature = {"device" : InputSignature(default_value="cpu"),
                       "logger" : InputSignature(),
                       "num_episodes" : InputSignature(),
                       "environment" : InputSignature(),
                       "state_memory_size" : InputSignature(),
                       "agents" : InputSignature(default_value={}),
                       "limit_steps" : InputSignature(),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100),
                       "rl_trainer" : InputSignature(default_value=''),
                       "created_agents_input" : InputSignature(
                            default_value={},
                            description='The input that will be passed to agents created by this pipeline')}
    
    exposed_values = {"total_steps" : 0} #this means we'll have a dic "values" with this starting values


    # INITIALIZATION ----------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
        self.lg = self.input["logger"]
        self.lg_profile = self.lg.createProfile(self.name)
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env = self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]
                
        self.agents = self.input["agents"] #this is a dictionary with {agentName -> agentComponent}, the environment must be able to return the agent name
        
        
        self.rl_trainer = self.input["rl_trainer"]
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        self.total_score = []
        
        self.episode_durations = []
        
        self.episode_time_per_step_durations = []
        
        self.configure_device(self.device)

        self.env.set_device(self.device)
        
        self.initialize_agents_components()
            
        if self.rl_trainer == '':
            self.initialize_trainer()
            
        for agent in self.agents.values(): #connect agents to rl_trainer
            agent.pass_input({"training_context" : self.rl_trainer.values}) 
        
        
    def configure_device(self, str_device_str):
        
        try:

            self.lg_profile.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg_profile.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg_profile.writeLine("The model will trained and evaluated on: " + str(self.device))
            

    def initialize_trainer(self):
        
        self.lg_profile.writeLine("Initializing trainer")
        
        rl_trainer_input = {
            "device" : "gpu",
            "logger" : self.lg,
            "num_episodes" : self.num_episodes,
            "state_memory_size" : self.state_memory_size,
            "environment" : self.env,
            "limit_steps" : self.limit_steps ,
            "optimization_interval" : self.optimization_interval,
            "agents" : self.agents
        }

        self.rl_trainer = RLTrainerComponent(input=rl_trainer_input)        

    def initialize_agents_components(self):

        if self.agents  == {}:
            self.create_agents()
        

    def create_agents(self):

        self.lg_profile.writeLine("Creating agents")

        agents = {}

        agentId = 1        
        for agent in self.env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {**self.input["created_agents_input"]} #the input of the agent, with base values defined by "agents_input"


            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name

            agent_logger = self.lg_profile.openChildLog(logName=agent_name)
            agent_input["logger"] = agent_logger

            state = self.env.observe(agent)
            self.lg_profile.writeLine("State for agent " + agent_name + " has shape: Z: " + str(len(state)) + " Y: " + str(len(state[0])) + " X: " + str(len(state[0][0])))

            z_input_size = len(state) * self.state_memory_size
            y_input_size = len(state[0])
            x_input_size = len(state[0][0])

            n_actions = self.env.action_space(agent).n
            print(f"Action space of agent {agent}: {self.env.action_space(agent)}")

            agent_input["model_input_shape"] = [x_input_size, y_input_size, z_input_size]
            agent_input["model_output_shape"] = n_actions
            
            agent_input["device"] = self.device       

            agents[agent] = AgentComponent(input=agent_input)

            self.lg_profile.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg_profile.writeLine("Initialized " + str(agents) + " agents")

        self.agents = agents    
        
        
    # TRAINING_PROCCESS ----------------------
    
    @requires_input_proccess
    def train(self):
        self.rl_trainer.run_episodes()