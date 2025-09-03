




from typing import Dict
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.loggers.result_logger import ResultLogger
from automl.rl.environment.environment_components import EnvironmentComponent

from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
import torch
import time

class AgentTrainer(ComponentWithLogging, ComponentWithResults):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                       "optimization_interval" : InputSignature(),
                       
                       "times_to_learn" : InputSignature(default_value=1), 
                       
                       "learning_start_ep_delay" : InputSignature(default_value=-1),
                       "save_interval" : InputSignature(default_value=100),
                        
                       "device" : InputSignature(get_from_parent=True, ignore_at_serialization=True),
                            
                        "agent" : ComponentInputSignature(),

                       "batch_size" : InputSignature(default_value=32),
                    
                       "discount_factor" : InputSignature(default_value=0.95),

                       "memory" : ComponentInputSignature(
                            default_component_definition=(TorchMemoryComponent, {}),
                       ),
                       
                       "learner" : ComponentInputSignature(
                            default_component_definition=(DeepQLearnerSchema, {})
                        )
                       
                       }
    
    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0,
                      } #this means we'll have a dic "values" with this starting values
    
    results_columns = ["episode", "episode_reward", "episode_steps", "avg_reward"]
    
    def __init__(self, *args, **kwargs): #Initialization done only when the object is instantiated
        super().__init__(*args, **kwargs)
        
        self.reset_training()


    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.optimization_interval = self.input["optimization_interval"]
        self.device = self.input["device"]
        self.save_interval = self.input["save_interval"]
    
        self.BATCH_SIZE = self.input["batch_size"] #the number of transitions sampled from the replay buffer
        self.discount_factor = self.input["discount_factor"] # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                              
        self.times_to_learn = self.input["times_to_learn"]    
        
        self._initialize_delays()                  
                                
        self.initialize_agent()
        self.initialize_learner()
        self.initialize_memory()
        self.initialize_temp()
                                

        

    # INITIALIZATION ---------------------------------------------
    
    def _initialize_delays(self):
        
        self.learning_start_ep_delay = self.input["learning_start_ep_delay"]
        
    
    def initialize_agent(self):
    
        self.agent : AgentSchema = ComponentInputSignature.get_component_from_input(self, "agent")
        self.agent.proccess_input_if_not_proccesd()
        
        
    def initialize_learner(self):
        
        self.learner : LearnerSchema = ComponentInputSignature.get_component_from_input(self, "learner")
        self.learner.pass_input({"device" : self.device, "agent" : self.agent})
        


    def initialize_memory(self):
        
        self.memory : MemoryComponent = ComponentInputSignature.get_component_from_input(self, "memory")
        
        self.memory_fields_shapes = [] # tuples of (name_of_field, dimension)
            
        self.memory.pass_input({
                                    "device" : self.device,
                                })
        
        
    def initialize_temp(self):
        
        self.state_memory_temp = self.agent.allocate_tensor_for_state()
        

        
    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    @requires_input_proccess
    def calculate_results(self):
        
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["total_steps"]]
            }
        
    
    # TRAINING_PROCESS ----------------------
    
    def reset_training(self):
        
        '''Resets the training values, such as the total number of episodes'''
        
        self.values["total_steps"] = 0
        self.values["episodes_done"] = 0
        
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        
        
        
    @requires_input_proccess
    def setup_training_session(self):
        
        self.lg.writeLine("Setting up training session")
        
        
                
    @requires_input_proccess
    def end_training(self):        
        self.lg.writeLine("Ending training session")
        
        
    
    @requires_input_proccess
    def setup_episode(self, env : EnvironmentComponent):
        
        self.lg.writeLine(f"Setting up episode {self.values['episodes_done'] + 1}", file=self.TRAIN_LOG)

        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
                
        self.agent.reset_agent_in_environment(env.observe(self.agent.name))
        
            
        
    @requires_input_proccess
    def end_episode(self):
        
        self.values["total_steps"] = self.values["total_steps"] + self.values["episode_steps"]
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        
        
        self.lg.writeLine("Ended episode: " + str(self.values["episodes_done"]) + " with duration: " + str(self.values["episode_steps"]) + ", total reward: " + str(self.values["episode_score"]), file=self.TRAIN_LOG)
        
        self.calculate_and_log_results()
    
    def _learn_if_needed(self):
        
        can_learn_by_ep_delay = self.learning_start_ep_delay < 1 or self.values["episodes_done"] >= self.learning_start_ep_delay
        
        if can_learn_by_ep_delay:          
        
            if self.values["total_steps"] % self.optimization_interval == 0:
                
                self.lg.writeLine(f"In episode (total) {self.values['episodes_done']}, optimizing at step {self.values['episode_steps']} that is the total step {self.values['total_steps']}", file=self.TRAIN_LOG)
                
                for _ in range(self.times_to_learn):
                    self.optimizeAgent()
            
        
    @requires_input_proccess
    def do_training_step(self, i_episode, env : EnvironmentComponent):
        
            '''
            Does a step, in which the agent acts and observers the transition
            Note that any other agents will not notice this change
            '''
        
            observation = env.observe(self.name)
            
            with torch.no_grad():                
                action = self._select_action(observation) # decides the next action to take (can be random)
                                                                                         
            env.step(action.item()) #makes the game proccess the action that was taken
                
            observation, reward, done, info = env.last()
                        
            self.values["episode_score"] = self.values["episode_score"] + reward
                            
            self._observe_transiction_to(observation, action, reward)
            
            self.values["episode_steps"] = self.values["episode_steps"] + 1
            self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step                                
            
            self._learn_if_needed() # uses the learning strategy to learn if it verifies the conditions to do so
                
            return reward, done
         
                            
        #if we reached a point where it is supposed to save
        #if(i_episode > 0 and i_episode < self.num_episodes - 1 and i_episode % self.save_interval == 0):
        #    self.lg.writeLine("Doing intermedian saving of results during training...", file=self.TRAIN_LOG)
        #    self.saveData()
        

    def _observe_transiction_to(self, new_state, action, reward):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''        
        
        raise NotImplementedError("This is not implemented in the base class")

        
        
    def observe_new_state(self, env : EnvironmentComponent):
        '''Makes the agent observe a new state, remembering it in case it needs that information in future computations'''
        self.agent.update_state_memory(env.observe(self.name))
        

    def _select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''

        return self.agent.policy_predict(state)



    def optimizeAgent(self):
        
        self.lg.writeLine("Optimizing agent...", file=self.TRAIN_LOG)
        
        timeBeforeOptimizing = time.time()
                            
        self.optimize_policy_model() # TODO : Take attention to this, the agents optimization strategy is too strict
        
        duration = time.time() - timeBeforeOptimizing
        
        self.lg.writeLine("Optimization took " + str(duration) + " seconds", file=self.TRAIN_LOG)
        
        
        
    def optimize_policy_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        #a batch of transitions [(state, action next_state, reward)] transposed to [ (all states), (all actions), (all next states), (all rewards) ]
        batch = self.memory.sample_transposed(self.BATCH_SIZE)
        
                
        self.learner.learn(batch, self.discount_factor)