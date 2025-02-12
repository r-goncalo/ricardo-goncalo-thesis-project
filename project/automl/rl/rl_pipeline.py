from automl.component import InputSignature, Schema, requires_input_proccess, uses_component_exception
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.model_components import ConvModelSchema
from automl.rl.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import PettingZooEnvironmentLoader
from automl.loggers.logger_component import LoggerSchema

import torch

# TODO this is missing the evaluation component on a RLPipeline
class RLPipelineComponent(LoggerSchema):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {"device" : InputSignature(default_value="cpu", ignore_at_serialization=True),
                       "num_episodes" : InputSignature(),
                       "environment" : InputSignature(generator= lambda self : self.initialize_child_component(PettingZooEnvironmentLoader)),
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
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env = self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]
                
        self.agents = self.input["agents"] #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name
        
        
        self.rl_trainer = self.input["rl_trainer"]
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        self.total_score = []
        
        self.episode_durations = []
        
        self.episode_time_per_step_durations = []
        
        self.configure_device(self.device)

        self.env.pass_input({"device" : self.device})
        
        self.initialize_agents_components()
            
        if self.rl_trainer == '':
            self.initialize_trainer()
            
        for agent in self.agents.values(): #connect agents to rl_trainer
            agent.pass_input({"training_context" : self.rl_trainer.values}) 
            
        
        
    def configure_device(self, str_device_str):
        
        try:

            self.lg.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg.writeLine("The model will trained and evaluated on: " + str(self.device))
            

    def initialize_trainer(self):
        
        self.lg.writeLine("Initializing trainer")
        
        rl_trainer_input = {
            "device" : self.device,
            "logger" : self.lg,
            "num_episodes" : self.num_episodes,
            "state_memory_size" : self.state_memory_size,
            "environment" : self.env,
            "limit_steps" : self.limit_steps ,
            "optimization_interval" : self.optimization_interval,
            "agents" : self.agents
        }

        self.rl_trainer = self.initialize_child_component(RLTrainerComponent, input=rl_trainer_input)        

    def initialize_agents_components(self):

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
            agent_input["logger"] = agent_logger

            state = self.env.observe(agent)
            self.lg.writeLine("State for agent " + agent_name + " has shape: Z: " + str(len(state)) + " Y: " + str(len(state[0])) + " X: " + str(len(state[0][0])))

            z_input_size = len(state) * self.state_memory_size
            y_input_size = len(state[0])
            x_input_size = len(state[0][0])

            n_actions = self.env.action_space(agent).n
            print(f"Action space of agent {agent}: {self.env.action_space(agent)}")

            agent_input["model_input_shape"] = [x_input_size, y_input_size, z_input_size]
            agent_input["model_output_shape"] = n_actions
            
            agent_input["device"] = self.device       

            agents[agent] = self.initialize_child_component(AgentSchema, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg.writeLine("Initialized " + str(agents) + " agents")

        self.agents = agents  
        self.input["agents"] = agents #this is done because we want to save these agents in the configuration
        


        
        
    # TRAINING_PROCCESS ----------------------
        
    @uses_component_exception
    @requires_input_proccess
    def train(self):        
        self.rl_trainer.run_episodes()