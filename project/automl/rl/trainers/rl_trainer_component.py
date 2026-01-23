
import math
from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentDictInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.aec_environment import AECEnvironmentComponent

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from automl.basic_components.exec_component import ExecComponent



class RLTrainerComponent(ComponentWithLogging, ComponentWithResults, ExecComponent):

    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                       "num_episodes" : InputSignature(default_value=-1, description="Number of episodes to do in this training session"),
                       "limit_total_steps" : InputSignature(default_value=-1, description="Number of total steps to do in this training session"), # Note how this changes with multiple agents
                       
                       "fraction_training_to_do" : InputSignature(mandatory=False),

                       "environment" : InputSignature(),
                       
                       "agents" : ComponentDictInputSignature(),
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

        self.lg.writeLine(f"Setting up RL trainer with initial values: {self.values}")
        
        self.device = self.get_input_value("device")
    
        self.limit_steps = self.get_input_value("limit_steps")

        self.agents_trainers_input = self.get_input_value("agents_trainers_input")

        self.default_trainer_class = self.get_input_value("default_trainer_class")
        
        self._initialize_limit_numbers()
        
        self.env : AECEnvironmentComponent = self.get_input_value("environment")
        
        self.setup_agents()

        self._optimizations_prediction()
        
        
    def _initialize_limit_numbers(self):
        
        self.limit_total_steps = self.get_input_value("limit_total_steps")
        self.num_episodes = self.get_input_value("num_episodes")  
        
        if self.limit_total_steps <= 0 and self.num_episodes <= 0:
            raise Exception("No stop condition defined")

        self._fraction_training_to_do = self.get_input_value("fraction_training_to_do")

        if self._fraction_training_to_do is None and self._times_to_run is not None:
            self._fraction_training_to_do = 1 / self._times_to_run

            self.lg.writeLine(f"As no fraction of training to do was specified, and times to run was specified ({self._times_to_run}), fraction of training is now 1 / {self._times_to_run}: {self._fraction_training_to_do}")


    
    def setup_agents(self):
        
        agents_in_input : Dict[str, AgentTrainer | AgentSchema] = self.get_input_value("agents", look_in_attribute_with_name="agents")
        
        self.agents_trainers : Dict[str, AgentTrainer] = self.values.get("agents_trainers", {})
        self.values["agents_trainers"] = self.agents_trainers
                
        self.agents_names_in_environment = self.env.agents()
        agents_names_in_environment = [*self.agents_names_in_environment]
        passed_agents_in_input = agents_in_input.keys()

        for key in passed_agents_in_input:

            if key in agents_names_in_environment:
                agents_names_in_environment.pop(agents_names_in_environment.index(key))

            else:
                raise Exception(f"Passed name for agent not in environment: {key}")
            
            agent_in_input = agents_in_input[key]

            agent_trainer_input = {**self.agents_trainers_input}
                
            if isinstance(agent_in_input, AgentSchema):
                
                self.lg.writeLine(f"Agent {key} came without a trainer in input")

                if key in self.agents_trainers.keys():
                    self.lg.writeLine(f"Agent {key} already had a loaded trainer")
                    agent_trainer = self.agents_trainers[key]
                    agent_trainer.pass_input(agent_trainer_input)

                else:
                    self.lg.writeLine(f"Agent {key} did not have a loaded trainer, creating one...")
                    agent_trainer_input_in_creation = {**agent_trainer_input, "agent" : agent_in_input} 

                    agent_trainer = self.initialize_child_component(self.default_trainer_class, agent_trainer_input_in_creation)

                    self.agents_trainers[key] = agent_trainer

                agents_in_input[key] = agent_trainer #puts the agent trainer in input too
    
            elif isinstance(agent_in_input, AgentTrainer):

                self.lg.writeLine(f"Agent {key} is already a trainer...")
                
                self.agents_trainers[key] = agent_in_input
                agent_in_input.pass_input(agent_trainer_input)

        self.input.pop("agents_trainers_input", None)

        



    def _make_optimization_prediction_for_agent_episodes(self, agent_key):
        raise NotImplementedError()
    
    def _make_optimization_prediction_for_agent_total_steps(self, agent_key):
        return self.agents_trainers[agent_key].make_optimization_prediction_for_agent_steps(self.limit_total_steps)
        

    def _make_optimization_prediction_for_agent(self, agent_key):

        if self.num_episodes > 1 and self.limit_total_steps > 1:
            raise Exception("Can't make prediction")
        
        elif self.num_episodes > 1:
            return self._make_optimization_prediction_for_agent_episodes(agent_key)

        elif self.limit_total_steps > 1:
            return self._make_optimization_prediction_for_agent_total_steps(agent_key)
        else:
            raise Exception("Can't make prediction")


    def _optimizations_prediction(self):

        '''Setup the predicted value for the optimizations to do'''

        self.predict_optimizations_to_do = self.get_input_value("predict_optimizations_to_do")

        if self.predict_optimizations_to_do:

            self.lg._writeLine("RLTrainer will try to predict the optimizations it has to do by agent")

            self.values["optimizations_to_do_per_agent"] = {}

            for key in self.agents_trainers:

                optimizations_for_agent = int(math.ceil(self._make_optimization_prediction_for_agent(key)))

                self.values["optimizations_to_do_per_agent"][key] = optimizations_for_agent
                self.lg._writeLine(f"RLTrainer predicted it will do {optimizations_for_agent} optimizations for agent with key '{key}'")





    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    def calculate_results(self):
                
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]],
            "total_steps" : [self.values["total_steps"]]
            }
    

    # TRAINING_PROCESS -------------------------------------------------------------------------------

    def _return_fraction_of_training_stop_value_if_any(self, training_stop_value):

        if self._fraction_training_to_do is not None:
            return training_stop_value * self._fraction_training_to_do

        else:
            return training_stop_value
        

    def _check_if_to_end_episode_by_steps_done(self):

        if self.limit_steps >= 1: # if we're using num episodes to stop training

            if self.values["episode_steps"] >= self.limit_steps:
                self.lg._writeLine("In episode " + str(self.values["episodes_done"]) + ", reached step " + str(self.values["episode_steps"]) + " that is beyond the current limit, " + str(self.limit_steps))
                return True
        
        return False


    def _check_if_to_end_episode(self):
        
        if self._check_if_to_end_episode_by_steps_done() or self._check_if_to_end_training_by_total_steps():
            return True
        
        return False
    
    
    def _check_if_to_end_training_by_episodes_done(self):

        if self.num_episodes >= 1: # if we're using num episodes to stop training

            max_episodes_to_do = self._return_fraction_of_training_stop_value_if_any(self.num_episodes)

            if self.values["episode_done_in_session"] >= max_episodes_to_do:
                self.lg._writeLine("Reached episode " + str(self.values["episodes_done"]) + " that is beyond the current limit, " + str(max_episodes_to_do))
                return True
        
        return False
    

    def _check_if_to_end_training_by_total_steps(self):

        if self.limit_total_steps >= 1: # if we're using steps to stop training

            max_total_steps_to_do = self._return_fraction_of_training_stop_value_if_any(self.limit_total_steps)

            if  self.values["steps_done_in_session"] >= max_total_steps_to_do:
                self.lg._writeLine(f"Total episodes done in this session, {self.values['steps_done_in_session']}, is greater than the limit for it, {max_total_steps_to_do}")
                return True
                
                
        return False
    

    def _check_if_to_end_training_session(self):

        if self._check_if_to_end_training_by_episodes_done() or self._check_if_to_end_training_by_total_steps():
            return True # we end the training

        return False
        


    @requires_input_proccess
    def run_episodes(self):
        
        '''
        Starts and runs training      
        '''

        self.setup_training_session()     

        self.lg.writeLine(f"Starting to run episodes with initial values: {self.values}")   

        while True: # loop of episodes and check end conditions
    
            self.run_single_episode(self.values["episodes_done"])
                
            if self._check_if_to_end_training_session():
                break

        self.end_training_session()



    def setup_training_session(self):

        self.lg._writeLine(f"Starting to run training with number of episodes: {self.num_episodes} and total step limit: {self.limit_total_steps}")

        if self._fraction_training_to_do != None:

            if self._fraction_training_to_do <= 0 or self._fraction_training_to_do > 1:
                raise Exception(f"Fraction of training to do must be between 0 and 1, was {self._fraction_training_to_do}")

            self.lg._writeLine(f"Only doing a fraction of {self._fraction_training_to_do} of the training")

        self.lg._writeLine(f"Resetting the environment...")

        self.env.total_reset()

        for agent_in_training in self.agents_trainers.values():
            agent_in_training.setup_training_session() 


        self.values["episode_done_in_session"] = 0
        self.values["steps_done_in_session"] = 0



    def end_training_session(self):

        self.lg._writeLine(f"Ended training with values: {self.values}")
        
        for agent_in_training in self.agents_trainers.values():
            agent_in_training.end_training()            
                
        self.env.close()

    
    def setup_single_episode(self, i_episode):

        self.env.reset()
        
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        
        for agent_in_training in self.agents_trainers.values():
            agent_in_training.setup_episode(self.env) 

    
    def run_episode_step_for_agent_name(self, i_episode, agent_name):

        agent_in_training = self.agents_trainers[agent_name] #gets the agent trainer for the current agent
            
        reward, done, truncated = agent_in_training.do_training_step(i_episode, self.env)
                        
        for other_agent_name in self.agents_trainers.keys(): #make the other agents observe the transiction without remembering it
            if other_agent_name != agent_name:
                    self.agents_trainers[other_agent_name].observe_new_state(self.env)
                    
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1
        self.values["steps_done_in_session"] = self.values["steps_done_in_session"] + 1
            
        self.values["episode_score"] = self.values["episode_score"] + reward

        return done, truncated
            

    
    def run_single_episode(self, i_episode):
                        
        self.setup_single_episode(i_episode)
                
        for agent_name in self.env.agent_iter(): #iterates infinitely over the agents that should be acting in the environment

            done, truncated = self.run_episode_step_for_agent_name(i_episode, agent_name)
                      
            if done or truncated or self._check_if_to_end_episode():
                break                      


        for agent_in_training in self.agents_trainers.values():
            agent_in_training.end_episode() 
        
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        self.values["episode_done_in_session"] = self.values["episode_done_in_session"] + 1
        
        self.calculate_and_log_results()



    def _algorithm(self):
        
        self.run_episodes() #trains the agents in the reinforcement learning pipeline
        
        

            
        
                   

                        