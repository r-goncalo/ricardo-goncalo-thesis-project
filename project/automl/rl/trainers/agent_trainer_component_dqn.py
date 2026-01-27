from automl.component import requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature

from automl.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer

import torch

from automl.utils.shapes_util import single_action_shape

class AgentTrainerDQN(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                        "exploration_strategy" : ComponentInputSignature(mandatory=False
                                                        ),
                       
                       }        
        

    # INITIALIZATION ---------------------------------------------
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.initialize_exploration_strategy()
        

    def initialize_exploration_strategy(self):
                
        if "exploration_strategy" in self.input.keys():
            
            self.exploration_strategy : ExplorationStrategySchema = self.get_input_value("exploration_strategy") 
            self.exploration_strategy.pass_input(input= {"training_context" : self}) #the exploration strategy has access to the same training context
        
            self.lg.writeLine(f"Using exploration strategy {self.exploration_strategy.name}")
            
        else:
            self.lg.writeLine(f"No exploration strategy defined")
            
            self.exploration_strategy = None

        

    def initialize_memory(self):
        
        super().initialize_memory()
        
        
        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("state", self.agent.model_input_shape), 
                                        ("action", single_action_shape(self.agent.get_policy().get_policy_output_shape()), torch.int64),
                                        ("next_state", self.agent.model_input_shape),
                                        ("reward", 1),
                                        ("done", 1)
                                    ]
            
        self.memory.pass_input({
                                    "transition_data" : self.memory_fields_shapes
                                })
        

    @requires_input_proccess
    def end_training(self):        
        super().end_training()
        
        if self.exploration_strategy is not None:
            self.lg.writeLine(f"Exploration strategy values: \n{self.exploration_strategy.values}\n")
        

    def _observe_transiction_to(self, new_state, action, reward, done):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''
        
        self.state_memory_temp.copy_(self.agent.get_current_state_in_memory())
        
        self.agent.update_state_memory(new_state)
        
        next_state_memory = self.agent.get_current_state_in_memory()
                
        self.memory.push({"state" : self.state_memory_temp, "action" : action, "next_state" : next_state_memory, "reward" : reward, "done" : done})
        

        

    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''


        if self.exploration_strategy is not None:
  
            return self.exploration_strategy.select_action(self.agent, state)  

        else:
            return super().select_action(state)
        

    def select_action_with_memory(self):

        if self.exploration_strategy is not None:
  
            return self.exploration_strategy.select_action_with_memory(self.agent)  

        else:
            return self.agent.policy_predict_with_memory()
        
        