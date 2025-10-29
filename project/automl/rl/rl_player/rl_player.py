import os
import traceback
from automl.basic_components.exec_component import ExecComponent
from automl.component import InputSignature, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.environment.environment_components import EnvironmentComponent
from automl.basic_components.state_management import StatefulComponent

from automl.rl.rl_setup_util import initialize_agents_components

from automl.utils.configuration_component_utils import save_configuration
from pyparsing import Dict
import torch

import gc

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automl.utils.random_utils import generate_seed, do_full_setup_of_seed

# TODO this is missing the evaluation component on a RLPipeline
class RLPlayer(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent):
    
    
    parameters_signature = {
                                                                                       
                       "environment" :  ComponentInputSignature(),
                       "agents" : InputSignature(),
                       "agents_input" : InputSignature(default_value={}),
                       "num_episodes" : InputSignature(default_value=1),
                       "limit_steps" : InputSignature(default_value=-1),
                       "store_env_at_end" : InputSignature(default_value=False)

                       
                       }
    

    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0
                      } 
    
    results_columns = ["episode", "episode_reward", "episode_steps", "avg_reward", "environment"]

    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()

        self.env : EnvironmentComponent = ComponentInputSignature.get_value_from_input(self, "environment")
        self.num_episodes = self.input["num_episodes"]
        
        self.limit_steps = self.input["limit_steps"]

        self.store_env_at_end = self.input["store_env_at_end"]

        self.__setup_agents()

        
    def __setup_agents(self):
        
        self.agents : Dict[str, AgentSchema] = initialize_agents_components(self.input["agents"], self.env, self.input["agents_input"], self)
        
        # if the agents have no base directory associated with it, use RL player's
        for agent in self.agents.values():
            if not "base_directory" in agent.input.keys():
                self.lg.writeLine(f"Agent {agent.name} has no base directory, passing player's directory to it")
                agent.pass_input({"base_directory" : self.get_artifact_directory()})
        
    
    def __setup_episode(self):

        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        self.env.reset()
        
        self.lg.writeLine("Starting episode " + str(self.values["episodes_done"] + 1) + " with agents: " + str(self.agents.keys()))
        
        self.lg.writeLine(f"The environment is: {self.env.name} with info: {self.env.get_env_info()}")
                
        for agent in self.agents.values():
            agent.reset_agent_in_environment(self.env.observe(agent.name))

    def __end_episode(self):
        self.values["total_steps"] = self.values["total_steps"] + self.values["episode_steps"]
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        
        self.lg.writeLine(f"Finished episode {self.values['episodes_done']} with results: {self.values}")
        
        self.calculate_and_log_results()


    def __run_episode(self):
        
        for agent_name in self.env.agent_iter():
            
            reward, done = self.__do_agent_step(agent_name)
            
            for other_agent_name in self.agents.keys(): #make the other agents observe the transiction without remembering it
                if other_agent_name != agent_name:
                    self.agents[other_agent_name].observe_new_state(self.env)
                            
            if done:
                break
            if self.limit_steps >= 1 and self.values["episode_steps"] >= self.limit_steps:
                self.lg.writeLine("In episode " + str(self.values["episodes_done"]) + ", reached step " + str(self.values["episode_steps"]) + " that is beyond the current limit, " + str(self.limit_steps))
                break
            
            
    def __do_agent_step(self, agent_name):
        
        agent : AgentSchema = self.agents[agent_name]
        
        observation = self.env.observe(agent_name)
        
        with torch.no_grad():                
            action = agent.policy_predict(observation) # decides the next action to take (can be random)
                
        self.env.step(action) #makes the game proccess the action that was taken
                
        observation, reward, done, info = self.env.last()
                        
        self.values["episode_score"] = self.values["episode_score"] + reward
                      
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step
        
        return reward, done
    
    
    @requires_input_proccess
    def play(self):
        
        for ep in range(self.num_episodes):

            self.__setup_episode()

            self.__run_episode()

            self.__end_episode()
            
        if self.store_env_at_end:
            save_configuration(self.env, self.get_artifact_directory(), "env_config.json", save_exposed_values=True, ignore_defaults=False)
            
                    
        self.env.close()


    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    def calculate_results(self):
                
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]],
            "environment" : [self.env.name]
            }
        



    def _algorithm(self):
        self.play()        
        
    

        
