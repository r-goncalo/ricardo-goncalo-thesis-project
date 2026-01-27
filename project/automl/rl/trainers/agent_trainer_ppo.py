from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.trainers.agent_trainer_component import AgentTrainer
import torch

from automl.utils.shapes_util import single_action_shape


class AgentTrainerPPO(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
                       
                       }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
                            
        if self.BATCH_SIZE != None:
            raise NotImplementedError("PPO Agent traienr stil does not have a sampling strategy implemented with batch size")
        

    # INITIALIZATION ---------------------------------------------
    

    def initialize_memory(self):
        
        super().initialize_memory()
                
        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("state", self.agent.model_input_shape), 
                                        ("action", single_action_shape(self.agent_policy.get_policy_output_shape()), torch.int64),
                                        ("next_state", self.agent.model_input_shape),
                                        ("reward", 1),
                                        ("done", 1),
                                        ("log_prob", 1) #log probability of chosing the stored action
                                    ]
            
        self.memory.pass_input({
                                    "device" : self.device,
                                    "transition_data" : self.memory_fields_shapes
                                })
        
        self.memory.clear() # IN PPO the memory must be filled only with transitions the current policy did


    def initialize_agent(self):
        
        super().initialize_agent()

        if not isinstance(self.agent_policy, StochasticPolicy):
            raise Exception("PPO trainer needs a stochastic policy")
        

        
    
    # TRAINING_PROCESS ---------------------
         

    def _observe_transiction_to(self, new_state, action, reward, done):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''
                
        self.state_memory_temp.copy_(self.agent.get_current_state_in_memory())
        
        self.agent.update_state_memory(new_state)
        
        next_state_memory = self.agent.get_current_state_in_memory()
                
        #we can push in this way because the pushed tensors are actually cloned into memory
        self.memory.push({"state" : self.state_memory_temp, "action" : action, "next_state" : next_state_memory, "reward" : reward, "log_prob" : self.last_log_prob, "done" : done})
               
        
    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''
                
        action, log_prob = self.agent.call_policy_method(self.agent_policy.predict_with_log, state) 
        
        self.last_log_prob = log_prob
        
        return action
        



    def optimizeAgent(self):

        '''Optimizes the agent for the specified number of times and then clears the memory'''
    
        super().optimizeAgent()
        
        self.memory.clear() # in PPO the policy must be filled only with transitions the current agent did       
