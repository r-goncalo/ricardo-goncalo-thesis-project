
from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.loggers.component_with_results import ComponentWithResults
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.environment_components import EnvironmentComponent

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging



class RLTrainerComponent(ComponentWithLogging, ComponentWithResults):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                       "num_episodes" : InputSignature(default_value=-1),
                       "limit_total_steps" : InputSignature(default_value=-1),
                       "environment" : InputSignature(),
                       
                       "agents" : InputSignature(),
                       "agents_trainers_input" : InputSignature(default_value={}),
                       "default_trainer_class" : InputSignature(default_value=AgentTrainer),
                       
                       "limit_steps" : InputSignature(default_value=-1),
                       "save_interval" : InputSignature(default_value=100)
                       
                       }
    
    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0,
                      "episode_done_in_session" : 0,
                      "steps_done_in_session" : 0
                      } #this means we'll have a dic "values" with this starting values
    
    results_columns = ["episode", "episode_steps", "avg_reward", "total_reward", "total_steps"]

    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super().proccess_input_internal()
        
        self.device = self.input["device"]
    
        self.limit_steps = self.input["limit_steps"]
        
        self._initialize_limit_numbers()
        
        self.env : EnvironmentComponent = self.input["environment"]
            
        self.save_interval = self.input["save_interval"]

        self.values["episodes_done"] = 0
        self.values["total_steps"] = 0
        
        self.setup_agents()
        
        
    def _initialize_limit_numbers(self):
        
        self.limit_total_steps = self.input["limit_total_steps"]
        self.num_episodes = self.input["num_episodes"]  
        
        if self.limit_total_steps <= 0 and self.num_episodes <= 0:
            raise Exception("No stop condition defined")
        
        
    
    def setup_agents(self):
        
        agents : Dict[str, AgentTrainer | AgentSchema] = self.input["agents"]
        
        self.agents_in_training : Dict[str, AgentTrainer] = {}
        
        for key in agents:
            
            agent_trainer_input = {**self.input["agents_trainers_input"]}
                
            if isinstance(agents[key], AgentSchema):
                
                self.lg.writeLine(f"Agent {key} came without a trainer, creating one...")
                
                agent_trainer_input_in_creation = {**agent_trainer_input, "agent" : agents[key]} 
                
                agent_trainer = self.initialize_child_component(self.input["default_trainer_class"], agent_trainer_input_in_creation)
                
                self.agents_in_training[key] = agent_trainer
                agents[key] = agent_trainer #puts the agent trainer in input too
    
            elif isinstance(agents[key], AgentTrainer):
                
                self.agents_in_training[key] = agents[key]
                self.agents_in_training[key].pass_input(agent_trainer_input)


    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    def calculate_results(self):
                
        return {
            "episode" : [self.values["episodes_done"]],
            "total_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]],
            "total_steps" : [self.values["total_steps"]]
            }
    

    # TRAINING_PROCESS -------------------------------------------------------------------------------

    def _check_if_to_end_episode(self):

        
        if self.limit_steps >= 1 and self.values["episode_steps"] >= self.limit_steps:
            
            self.lg.writeLine("In episode " + str(self.values["episodes_done"]) + ", reached step " + str(self.values["episode_steps"]) + " that is beyond the current limit, " + str(self.limit_steps))
            return True
            
        if self.limit_total_steps >= 1 and self.values["steps_done_in_session"] >= self.limit_total_steps:
            return True
            
        
        return False
    

    def _check_if_to_end_training_session(self):

        
        if self.num_episodes >= 1 and self.input["episode_done_in_session"] >= self.num_episodes:
            self.lg.writeLine("Reached episode " + str(self.values["episodes_done"]) + " that is beyond the current limit, " + str(self.num_episodes))
            return True
        
        if self.limit_total_steps >= 1 and self.values["steps_done_in_session"] >= self.limit_total_steps:
            return True
        
        return False


    @requires_input_proccess
    def run_episodes(self):
        
        '''Starts training
        
           Args:
           
            :n_episodes, if defined, limits the number of episodes that will be done
        
        '''
        
        self.lg.writeLine(f"Starting to run {self.num_episodes} episodes of training")
        
            
        for agent_in_training in self.agents_in_training.values():
            agent_in_training.setup_training_session() 
            
            
        self.values["episode_done_in_session"] = 0
        self.values["steps_done_in_session"] = 0
        
        while True: # loop of episodes and check end conditions
            
            self.__run_episode(self.values["episodes_done"])
            
            for agent_in_training in self.agents_in_training.values():
                agent_in_training.end_episode() 
            
            self.values["episodes_done"] = self.values["episodes_done"] + 1
            
            self.calculate_and_log_results()
            
            self.values["episode_done_in_session"]
            
            if self._check_if_to_end_training_session():
                break
        
                
            
        for agent_in_training in self.agents_in_training.values():
            agent_in_training.end_training()            
                
        self.env.close()
            
    
    def __run_episode(self, i_episode):
                        
        self.env.reset()
        
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        
        for agent_in_training in self.agents_in_training.values():
            agent_in_training.setup_episode(self.env) 
            
                
        for agent_name in self.env.agent_iter(): #iterates infinitely over the agents that should be acting in the environment
                                            
            agent_in_training = self.agents_in_training[agent_name] #gets the agent trainer for the current agent
            
            reward, done = agent_in_training.do_training_step(i_episode, self.env)
                        
            for other_agent_name in self.agents_in_training.keys(): #make the other agents observe the transiction without remembering it
                if other_agent_name != agent_name:
                    self.agents_in_training[other_agent_name].observe_new_state(self.env)
                    
            self.values["episode_steps"] = self.values["episode_steps"] + 1
            self.values["total_steps"] = self.values["total_steps"] + 1
            self.values["steps_done_in_session"] = self.values["steps_done_in_session"] + 1
            
            self.values["episode_score"] = self.values["episode_score"] + reward
            
            if done or self._check_if_to_end_episode():
                break
            
        
                   

                        