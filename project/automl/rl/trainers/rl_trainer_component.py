
from typing import Dict
from automl.component import InputSignature, Schema, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.environment_components import EnvironmentComponent


class RLTrainerComponent(LoggerSchema):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
                        "device" : InputSignature(ignore_at_serialization=True),
                       "num_episodes" : InputSignature(),
                       "environment" : InputSignature(),
                       "agents" : InputSignature(),
                       "limit_steps" : InputSignature(),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100)}
    
    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0
                      } #this means we'll have a dic "values" with this starting values

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes = self.input["num_episodes"]  
        
        self.env : EnvironmentComponent = self.input["environment"]
        
        self.optimization_interval = self.input["optimization_interval"]
    
        self.save_interval = self.input["save_interval"]

        self.values["episodes_done"] = 0
        self.values["total_steps"] = 0
        
        self.result_logger = ResultLogger({ 
            "logger_object" : self.lg,
            "keys" : ["episode", "episode_steps", "avg_reward", "total_reward"]
            })
                
        self.setup_agents()
        
        
    
    def setup_agents(self):
        
        agents : Dict[str, AgentTrainer | AgentSchema] = self.input["agents"]
        
        self.agents_in_training : Dict[str, AgentTrainer] = {}
        
        for key in agents:
    
            if isinstance(agents[key], AgentSchema):
                
                self.lg.writeLine(f"Agent {key} came without a trainer, creating one...")
                
                agent_trainer_input = {"agent" : agents[key], "optimization_interval" : self.optimization_interval, "logger_object" : agents[key].lg} 
                
                agent_trainer = self.initialize_child_component(AgentTrainer, agent_trainer_input)
                
                self.agents_in_training[key] = agent_trainer
                agents[key] = agent_trainer #puts the agent trainer in input too
    
            elif isinstance(agents[key], AgentTrainer):
                
                self.agents_in_training[key] = agents[key]
                self.agents_in_training[key].pass_input({"logger_object" : agents[key].lg})



    # TRAINING_PROCESS -------------------------------------------------------------------------------


    @requires_input_proccess
    def run_episodes(self, n_episodes : int = None):
        
        '''Starts training
        
           Args:
           
            :n_episodes, if defined, limits the number of episodes that will be done
        
        '''
        
        self.lg.writeLine("Starting to run episodes of training")
        
            
        for agent_in_training in self.agents_in_training.values():
            agent_in_training.setup_training() 
            
        
        n_episodes_to_do = self.num_episodes - self.values["episodes_done"] if n_episodes == None else min(self.num_episodes - self.values["episodes_done"], n_episodes)
            
        #each episode is an instance of playing the game
        for _ in range(n_episodes_to_do):
            
            self.__run_episode(self.values["episodes_done"])
            
            for agent_in_training in self.agents_in_training.values():
                agent_in_training.end_episode() 
            
            self.values["episodes_done"] = self.values["episodes_done"] + 1
            
            self.result_logger.log_results({
                "episode" : [self.values["episodes_done"]],
                "total_reward" : [self.values["episode_score"]],
                "episode_steps" : [self.values["episode_steps"]], 
                "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]]
            })   

            ["episode", "episode_steps", "episode_reward", "episode_avg_reward", "avg_reward", "total_reward"]
            
                
            
        for agent_in_training in self.agents_in_training.values():
            agent_in_training.end_training()            
        
        self.result_logger.save_dataframe()
        
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
            self.values["episode_score"] = self.values["episode_score"] + reward
                            
            if done:
                break
            if self.limit_steps >= 1 and self.values["episode_steps"] >= self.limit_steps:
                self.lg.writeLine("In episode " + str(self.values["episodes_done"]) + ", reached step " + str(self.values["episode_steps"]) + " that is beyond the current limit, " + str(self.limit_steps))
                break
            
            
            
    def plot_results_graph(self):
           
       self.result_logger.plot_graph("episode", ["total_reward"])
       
       self.result_logger.plot_graph("episode", ["episode_steps"])
       
       self.result_logger.plot_graph("episode", ["avg_reward"])
        
                   
    def get_last_results(self):
        
        return self.result_logger.get_last_results()
    
    def get_results_logger(self) -> ResultLogger:
        
        return self.result_logger
                        