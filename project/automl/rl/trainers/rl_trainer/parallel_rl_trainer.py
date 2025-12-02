
import math
from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.loggers.component_with_results import ComponentWithResults
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.loggers.result_logger import ResultLogger

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent

from automl.rl.environment.parallel_environment import ParallelEnvironmentComponent
import torch

class RLTrainerComponentParallel(RLTrainerComponent):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                       "num_episodes" : InputSignature(default_value=-1, description="Number of episodes to do in this training session"),
                       "limit_total_steps" : InputSignature(default_value=-1, description="Number of total steps to do in this training session"), # Note how this changes with multiple agents
                       
                       "fraction_training_to_do" : InputSignature(mandatory=False),

                       "environment" : InputSignature(),
                       
                       "agents" : InputSignature(),
                       "agents_trainers_input" : InputSignature(default_value={}, ignore_at_serialization=True),
                       "default_trainer_class" : InputSignature(default_value=AgentTrainer),
                       
                       "limit_steps" : InputSignature(default_value=-1),

                       "predict_optimizations_to_do" : InputSignature(default_value=False),
                       
                       }
    
    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0,
                      "episode_done_in_session" : 0,
                      "steps_done_in_session" : 0
                      } #this means we'll have a dic "values" with this starting values
    
    results_columns = ["episode", "episode_steps", "avg_reward", "episode_reward", "total_steps"]

    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()

        if not isinstance(self.env, ParallelEnvironmentComponent):
            raise Exception(f"Parallel RL training requires parallel component")
        
        self.chosen_actions = {}
                                                                                     

    
    def choose_actions_for_agents(self, agents_names : list[str], i_episode):

        for agent_name in agents_names:

            agent_trainer = self.agents_in_training[agent_name]
        
            with torch.no_grad():                
                action = agent_trainer.select_action_with_memory() # decides the next action to take (can be random)

            self.chosen_actions[agent_name] = action.item()

        return self.chosen_actions
    


    def proccess_env_step_for_agents(self, i_episode, agents_names : list[str], actions, observations, rewards, terminations, truncations):

        done = False

        for agent_name in agents_names:
            
            agent_trainer = self.agents_in_training[agent_name]

            action = actions[agent_name]
            observation = observations[agent_name]
            reward = rewards[agent_name]
            termination = terminations[agent_name]
            truncation = truncations[agent_name]

            agent_trainer.do_after_training_step(i_episode, action, observation, reward, termination, truncation)

            done = done or termination or truncation

        return done


    
    def run_single_episode(self, i_episode):
                        
        self.setup_single_episode(i_episode)

        agent_names = self.env.parallel_agents() 

        while agent_names:

            actions = self.choose_actions_for_agents(agent_names, i_episode)

            observations, rewards, terminations, truncations, infos = self.env.step(actions)

            done = self.proccess_env_step_for_agents(i_episode, agent_names, actions, observations, rewards, terminations, truncations)

            self.values["episode_steps"] = self.values["episode_steps"] + 1
            self.values["total_steps"] = self.values["total_steps"] + 1
            self.values["steps_done_in_session"] = self.values["steps_done_in_session"] + 1

            self.values["episode_score"] = self.values["episode_score"] + sum(rewards.values())


            if done or self._check_if_to_end_episode():
                break    

            agent_names = self.env.parallel_agents()              

        for agent_in_training in self.agents_in_training.values():
            agent_in_training.end_episode() 
        
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        self.values["episode_done_in_session"] = self.values["episode_done_in_session"] + 1
        
        self.calculate_and_log_results()
            
        
                   

                        