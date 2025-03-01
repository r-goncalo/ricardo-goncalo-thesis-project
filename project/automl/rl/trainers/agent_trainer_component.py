




from typing import Dict
from automl.component import InputSignature, Schema, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema
from automl.rl.agent.agent_components import AgentSchema
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.environment_components import EnvironmentComponent

import torch
import time

class AgentTrainer(Schema):
    
    '''Describes a trainer specific for an agent, mostly used to control its logging'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
                       "agent" : InputSignature(possible_types=[AgentSchema]),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100)}
    
    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "total_score" : 0,
                      "episode_score" : 0
                      } #this means we'll have a dic "values" with this starting values

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
                                
        self.agent : AgentSchema = self.input["agent"]
        
        self.agent.pass_input({"training_context" : self})
        
        self.agent.proccess_input()
        
        self.lg = self.agent.lg.createProfile(object_with_name=self)
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        
        self.result_logger = ResultLogger({ "logger_object" : self.lg,
            "keys" : ["episode", "total_reward", "episode_steps", "avg_reward"]})
                    
    
    # TRAINING_PROCESS ---------------------
    
    @requires_input_proccess
    def setup_training(self):
        
        self.lg.writeLine("Setting up training session", file=self.TRAIN_LOG)
        
        self.values["total_steps"] = 0
        self.values["total_score"] = 0
        self.values["episodes_done"] = 0
                
    @requires_input_proccess
    def end_training(self):
        self.result_logger.save_dataframe()
        
        self.lg.writeLine(f"Values of exploraion strategy: {self.agent.exploration_strategy.values}", file=self.TRAIN_LOG)
    
    @requires_input_proccess
    def setup_episode(self, env : EnvironmentComponent):
        
        self.lg.writeLine(f"Setting up episode {self.values['episodes_done'] + 1}", file=self.TRAIN_LOG)

        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
                
        self.agent.reset_state_memory(env.observe(self.agent.name))
            
        
    @requires_input_proccess
    def end_episode(self):
        
        self.values["total_steps"] = self.values["total_steps"] + self.values["episode_steps"]
        self.values["total_score"] = self.values["total_score"] +  self.values["episode_score"]
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        
        
        self.lg.writeLine("Ended episode: " + str(self.values["episodes_done"]) + " with duration: " + str(self.values["episode_steps"]) + ", total reward: " + str(self.values["episode_score"]), file=self.TRAIN_LOG)
        
        self.result_logger.log_results({
            "episode" : [self.values["episodes_done"]],
            "total_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]]
            })  
        
        
    @requires_input_proccess
    def do_training_step(self, i_episode, env : EnvironmentComponent):
        
            observation = env.observe(self.name)
                                
            action = self.agent.select_action(observation) # decides the next action to take (can be random)
                                                     
            env.step(action) #makes the game proccess the action that was taken
                
            observation, reward, done, info = env.last()
            
            self.values["episode_score"] = self.values["episode_score"] + reward
                
            self.agent.observe_transiction_to(observation, action, reward)
                    
            self.values["episode_steps"] = self.values["episode_steps"] + 1
            self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step                                
            
            if self.values["total_steps"] % self.optimization_interval == 0:
                
                self.lg.writeLine(f"In episode {i_episode}, optimizing at step {self.values['episode_steps']} that is the total step {self.values['total_steps']}", file=self.TRAIN_LOG)
                self.optimizeAgent()
                
            return reward, done
         
                            
        #if we reached a point where it is supposed to save
        #if(i_episode > 0 and i_episode < self.num_episodes - 1 and i_episode % self.save_interval == 0):
        #    self.lg.writeLine("Doing intermedian saving of results during training...", file=self.TRAIN_LOG)
        #    self.saveData()
        
    def observe_new_state(self, env : EnvironmentComponent):
        '''Makes the agent observe a new state, remembering it in case it needs that information in future computations'''
        self.agent.observe_new_state(env.observe(self.name))


    def optimizeAgent(self):
        
        self.lg.writeLine("Optimizing agent...", file=self.TRAIN_LOG)
        
        timeBeforeOptimizing = time.time()
                            
        self.agent.optimize_policy_model() # TODO : Take attention to this, the agents optimization strategy is too strict
        
        duration = time.time() - timeBeforeOptimizing
        
        self.lg.writeLine("Optimization took " + str(duration) + " seconds", file=self.TRAIN_LOG)