from automl.basic_components.exec_component import ExecComponent
from automl.component import ParameterSignature, requires_input_proccess
from automl.core.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automl.fundamentals.acessories import AcessoryComponent
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.memory_samplers.memory_sampler import MemorySampler
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.environment.aec_environment import AECEnvironmentComponent

from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.policy.policy import Policy
import torch

from automl.basic_components.EventfulComponent import EventfulComponent, Event


class AgentTrainer(ComponentWithLogging, ComponentWithResults, EventfulComponent, ExecComponent):
    
    '''
    Describes a trainer specific for an agent, using a learner algorithm, memory and more

    It serves as a connection between an RLTrainer (which describes a general training algorithm for single or multi agent RL) and the agents

    It is resposible for knowing when the agent is learning or just being used, logging results, connecting the memory and the learning algorithm and so on

    Functionality such as using acessories is turned off when the agent is not learning
    '''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                       "optimization_interval" : ParameterSignature(description="How many steps between optimizations", default_value=1,
                                                                custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 100, "high": 500 }]}
                                                                ),
                       
                       "times_to_learn" : ParameterSignature(default_value=1, description="How many times to optimize at learning time",
                                                         custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 1, "high": 256 }]}), 
                       
                       "learning_start_ep_delay" : ParameterSignature(default_value=-1),
                        "learning_start_step_delay" : ParameterSignature(default_value=-1),

                       "save_interval" : ParameterSignature(default_value=100),
                        
                       "device" : ParameterSignature(get_from_parent=True, ignore_at_serialization=True),
                            
                        "agent" : ComponentParameterSignature(),

                       "batch_size" : ParameterSignature(mandatory=False, custom_dict={"hyperparameter_suggestion" : [ "cat", {"choices": [8, 16, 32, 64, 128, 256]}]}),
                       
                       "learn_with_all_memory" : ParameterSignature(default_value=False, 
                                                                description="When true, each learning will consist of dividing the entire memory into batches with the specified size, and learning with each of them"
                                                                ),
                    
                       "discount_factor" : ParameterSignature(
                           default_value=0.95,
                           custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.7, "high": 0.99999 }]}
                           ),

                       "memory" : ComponentParameterSignature(
                            default_component_definition=(TorchMemoryComponent, {}),
                       ),

                       "memory_transformer" : ComponentParameterSignature(mandatory=False),
                       
                       "learner" : ComponentParameterSignature(
                            default_component_definition=(DeepQLearnerSchema, {})
                        ),

                        "agent_trainer_acessories" : ComponentListParameterSignature(mandatory=False, description="Acessories used for when the agent is learning"),

                       "limit_steps" : ParameterSignature(
                           default_value=-1,
                           description="Limits the steps in a single training session"
                          ),                         

                       }
    
    exposed_values = {
                      "total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0,
                      "optimizations_done" : 0,
                      "average_optimization" : 0,
                      "external_end_requests" : None,
                      "is_training" : False
                      } #this means we'll have a dic "values" with this starting values
    
    STATIC_EVENTS = {
        "ended_agent_training" : Event()
        }
    
    results_columns = ["episode", "episode_reward", "episode_steps", "avg_reward", "is_training"]
    
    def __init__(self, *args, **kwargs): #Initialization done only when the object is instantiated
        super().__init__(*args, **kwargs)

        self.values["external_end_requests"] = {} # this is to be sure that same instances are not shared
        

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.optimization_interval = self.get_input_value("optimization_interval")
        self.device = self.get_input_value("device")
        self.save_interval = self.get_input_value("save_interval")
    
        self.BATCH_SIZE = self.get_input_value("batch_size") #the number of transitions sampled from the replay buffer
        
        self.learn_with_all_memory = self.get_input_value("learn_with_all_memory")
        
        self.discount_factor = self.get_input_value("discount_factor") # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                              
        self.times_to_learn = self.get_input_value("times_to_learn")

        self.limit_steps = self.get_input_value("limit_steps")

        if self.limit_steps >= 0:
            self.lg.writeLine(f"Will limit the learning of this agent to {self.limit_steps} steps in this training session")    
        
        self._initialize_delays()                  
                                
        self.initialize_agent()
        self.initialize_learner()
        self.initialize_memory()
        self.initialize_temp()
        self._initialize_acessories()

        self.values['is_training'] = False
                                

        

    # INITIALIZATION ---------------------------------------------

    def _initialize_acessories(self):
        self.agent_trainer_acessories : list[AcessoryComponent] = self.get_input_value("agent_trainer_acessories")

        if self.agent_trainer_acessories is None:
            self.agent_trainer_acessories : list[AcessoryComponent] = []

        else:
            for acessory in self.agent_trainer_acessories:
                acessory.pass_input({"affected_component" : self})
                self.lg.writeLine(f"Agent trainer has acessory: {acessory.name}")

    def _initialize_delays(self):
        
        self.learning_start_ep_delay = self.get_input_value("learning_start_ep_delay")
        self.learning_start_step_delay = self.get_input_value("learning_start_step_delay")
        
    
    def initialize_agent(self):
    
        self.agent : AgentSchema = self.get_input_value("agent", look_in_attribute_with_name="agent")
        self.agent.proccess_input_if_not_processed()

        self.agent_policy : Policy = self.agent.policy
        
        
    def initialize_learner(self):
        
        self.learner : LearnerSchema = self.get_input_value("learner", look_in_attribute_with_name="learner")
        self.learner.pass_input({"device" : self.device, "agent" : self.agent, "agent_trainer" : self})
        


    def initialize_memory(self):
        
        self.memory : MemoryComponent = self.get_input_value("memory", look_in_attribute_with_name="memory")
        
        self.memory_fields_shapes = [] # tuples of (name_of_field, dimension)
            
        self.memory.pass_input({
                                    "device" : self.device,
                                })
        
        self.memory_transformer : MemorySampler = self.get_input_value("memory_transformer")

        if self.memory_transformer is not None:
            self.lg.writeLine(f"Will use memory transformer: {self.memory_transformer.name}")
        
    def initialize_temp(self):
        
        self.state_memory_temp = self.agent.allocate_tensor_for_state()
        

        
    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    @requires_input_proccess
    def calculate_results(self):
        
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [self.values["episode_score"]],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [self.values["episode_score"] / self.values["episode_steps"]],
            "is_training" : [self.values['is_training']]
            }
        

    # PREDICTING OPTIMIZATIONS ------------------------------------------------

    @requires_input_proccess
    def make_optimization_prediction_for_agent_steps(self, total_steps):
        
        return  ( total_steps / self.optimization_interval ) * self.times_to_learn
    
    
    # STOP CONDITIONS ------------------------

    def _check_if_to_end_training_by_steps(self):

        if self.limit_steps >= 1: # if we're using steps to stop training

            if  self.values["steps_done_in_session"] >= self.limit_steps:
                self.lg._writeLine(f"Total episodes done in this session, {self.values['steps_done_in_session']}, is greater than the limit for it, {self.limit_steps}")
                return True                
                
        return False

    def _check_if_to_end_training(self):
        return self._check_if_to_end_training_by_steps()
    
    # TRAINING_PROCESS ----------------------


        
    def is_agent_training(self):
        return self.values['is_training']
        
        
    @requires_input_proccess
    def _setup_training_session(self):
        
        self.lg.writeLine("Setting up training session")

        self.values['is_training'] = True

        external_end_requests : dict = self.values["external_end_requests"]
        
        for external_end_key in external_end_requests:
            self.initialize_external_end_request(external_end_key)

        self.values["steps_done_in_session"] = 0

        self.start_algorithm() # this tells the compon
        
        
                
    @requires_input_proccess
    def end_training(self):

        if self.values['is_training']:
            self.lg.writeLine("Ending training session... (Note that the trainer can still be used)")

            self.values['is_training'] = False

            self.EVENTS["ended_agent_training"].notify(self.name)

            self.end_algorithm()
        
    
    def _is_over(self):
        
        isover = super()._is_over()

        # the training is considered over when the reason for the training ended was external
        if not isover:
            if len(self.values["external_end_requests"]) > 0 and all(self.values["external_end_requests"].values()):
                self.lg.writeLine(f"Training is considered over because all external conditions asked for the training of the agent to end")
                isover = True

        return isover
        
    
    @requires_input_proccess
    def setup_episode(self, env : AECEnvironmentComponent):
        
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        
        self.agent.reset_agent_in_environment(env.observe(self.agent.name))

        
            
        
    @requires_input_proccess
    def end_episode(self):
        
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        self.calculate_and_log_results()

        # we just log results when the agent is training
        if self.values['is_training']: 

            for acessory in self.agent_trainer_acessories:
                acessory.as_fun()

    
    # LEARNING AND INTERACTING WITH THE ENVIRONMENT -------------------------------
                
            
        
    @requires_input_proccess
    def do_training_step(self, i_episode, env : AECEnvironmentComponent):
        
            '''
            Does a step, in which the agent acts and observers the transition
            Note that any other agents will not notice this change without outside coordination
            
            This method can be used when the agent is not training
            '''
                    
            observation = env.observe(self.agent.name)
            
            with torch.no_grad():                
                action = self.select_action(observation).squeeze(0) # decides the next action to take (can be random)
                                                                                         
            env.step(action) #makes the game proccess the action that was taken
                
            observation, reward, done, truncated, info = env.last()
                        
            self.do_after_training_step(i_episode, action, observation, reward, done, truncated)
                
            return reward, done, truncated
             
                            
    def do_after_training_step(self, i_episode, action, observation, reward, done, truncated):

        '''
        Does the normal computation after a training step
        This assumes an observation was just made, an action chosen and the environment was notified of that action
        
        It observeses the transition of the environment to the received observation, and notes that an episode step was just made
        '''

        self.values["episode_score"] = self.values["episode_score"] + reward
                            
        self._observe_transiction_to(observation, action, reward, done)
            
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step      ~

        if self.values['is_training']:

            self.values["steps_done_in_session"] = self.values["steps_done_in_session"] + 1
            
            if self._check_if_to_end_training():
                self.end_training()

        self._learn_if_needed() # uses the learning strategy to learn if it verifies the conditions to do so
                
        

    def _observe_transiction_to(self, new_state, action, reward, done):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''        
        
        raise NotImplementedError("This is not implemented in the base class")

        
        
    def observe_new_state(self, env : AECEnvironmentComponent):
        '''Makes the agent observe a new state, remembering it in case it needs that information in future computations'''
        self.agent.update_state_memory(env.observe(self.agent.name))
        

    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''

        return self.agent.policy_predict(state)
    

    def select_action_with_memory(self):
        
        return self.agent.policy_predict_with_memory()

    # LEARNING PROCESS --------------------------------------------


    def _learn_if_needed(self):

        '''
        Does a learning step if the agent is still learning
        This means calling optimize agent          
        '''

        if self.values['is_training']:
        
            can_learn_by_ep_delay = self.learning_start_ep_delay < 1 or self.values["episodes_done"] >= self.learning_start_ep_delay
            can_learn_by_step_delay = self.learning_start_step_delay < 1 or self.values["total_steps"] >= self.learning_start_step_delay

            if can_learn_by_ep_delay and can_learn_by_step_delay:          
            
                if self.values["total_steps"] % self.optimization_interval == 0:

                    self.optimizeAgent()


    def optimizeAgent(self):

        '''Optimizes the trained agent for the number of specified times''' 

        if self.memory_transformer is not None:
            self.memory_transformer.prepare(self.memory)

        for _ in range(self.times_to_learn):
            self._optimize_policy_model() 
            self.values["optimizations_done"] += 1
        
        
    def _optimize_policy_model_with_batch(self, batch):
        self.learner.learn(batch, self.discount_factor)

        
    def _optimize_policy_model(self):

        sampler = self.memory if self.memory_transformer is None else self.memory_transformer

        if self.BATCH_SIZE != None:
        
            if len(self.memory) < self.BATCH_SIZE:
                return

            if self.learn_with_all_memory:
                batches = sampler.sample_all_with_batches(self. BATCH_SIZE)
            
            else:
                batches = [sampler.sample(self.BATCH_SIZE)]

        else:
            batches = [sampler.get_all()]        
                
        for b in batches:
            self._optimize_policy_model_with_batch(b)

    # EXTERNAL ACESSORIES PROCESSING ----------------------------------------------

    def request_end_from_external(self, key):

        '''
        Receives a request from an outside entity to terminate the training of this agent
        This usually means convergence was detected
        '''

        external_end_requests : dict = self.values["external_end_requests"]
        external_end_requests[key] = True

        if all(external_end_requests.values()):
            self.lg.writeLine(f"All external end requests: {external_end_requests} are true, requesting to end training...")
            self.end_training()

    def request_continue_from_external(self, key):

        external_end_requests : dict = self.values["external_end_requests"]
        external_end_requests[key] = False


    def initialize_external_end_request(self, key):

        self.lg.writeLine(f"Received request to register an end condition with key: {key}")

        external_end_requests = self.values["external_end_requests"]
        external_end_requests[key] = False