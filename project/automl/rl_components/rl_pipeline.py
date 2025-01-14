from ..component import input_signature, Component, requires_input_proccess
from .agent_components import AgentComponent
from .optimizer_components import AdamOptimizer
from .exploration_strategy_components import EpsilonGreedyStrategy
from .model_components import ConvModelComponent
from rl_trainer_component import RLTrainerComponent

import torch
import time

class RLTrainerPipeline(Component):

    TRAIN_LOG = 'train.txt'
    
    input_signature = {"device" : input_signature(),
                       "logger" : input_signature(),
                       "num_episodes" : input_signature(),
                       "environment" : input_signature(),
                       "state_memory_size" : input_signature(),
                       "agents" : input_signature(),
                       "limit_steps" : input_signature(),
                       "optimization_interval" : input_signature(),
                       "save_interval" : input_signature(default_value=100)}
    
    exposed_values = {"total_steps" : 0} #this means we'll have a dic "values" with this starting values

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
        self.lg = self.input["logger"]
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env = self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]
                
        self.agents = self.input["agents"] #this is a dictionary with {agentName -> agentComponent}, the environment must be able to return the agent name
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        self.total_score = []
        
        self.episode_durations = []
        
        self.episode_time_per_step_durations = []

    def initialize_agents_components(lg, env, learning_rate=0.001, state_memory_size=1, agents_input = {}):

        agents = {}

        agentId = 1        
        for agent in env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {**agents_input} #the input of the agent, with base values defined by "agents_input"


            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name

            agent_logger = lg.openChildLog(logName=agent_name)
            agent_input["logger"] = agent_logger

            state = env.observe(agent)
            lg.writeLine("State for agent " + agent_name + " has shape: Z: " + str(len(state)) + " Y: " + str(len(state[0])) + " X: " + str(len(state[0][0])))

            z_input_size = len(state) * state_memory_size
            y_input_size = len(state[0])
            x_input_size = len(state[0][0])

            n_actions = env.action_space(agent).n
            print(f"Action spac of agent {agent}: {env.action_space(agent)}")

            agent_model = ConvModelComponent(input={"board_x" : x_input_size, "board_y" : y_input_size, "board_z" : z_input_size, "output_size" : n_actions})
            agent_input["policy_model"] = agent_model
            agent_model.proccess_input()  

            agent_opimizer = AdamOptimizer( {"learning_rate" : learning_rate } )
            agent_input["optimizer"] = agent_opimizer         

            agents[agent] = AgentComponent(input=agent_input)

            lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        lg.writeLine("Initialized " + str(agents) + " agents")

        return agents