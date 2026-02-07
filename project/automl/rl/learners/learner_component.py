from automl.component import Component, InputSignature, requires_input_proccess


from automl.ml.memory.memory_utils import interpret_unit_values, interpret_values
from automl.rl.agent.agent_components import AgentSchema
import torch

class LearnerSchema(Component):
        
    parameters_signature = {
        "agent" : InputSignature(),
        "optimizations_per_learn" : InputSignature(default_value=1,custom_dict={
                                    "hyperparameter_suggestion" : ("int", {"low" : 1, "high" : 32})
                                })

    }
        
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
        self.agent : AgentSchema = self.get_input_value("agent")

        self.optimizations_per_learn : int = self.get_input_value("optimizations_per_learn")

    
    def _learn(self, trajectory, discount_factor):
        pass
        
    @requires_input_proccess
    def learn(self, trajectory, discount_factor) -> None:
        
        '''
            Learns the policy using the current policy and the trajectory it is supposed to learn
            
            Remember that On-Policy algorithms expect the trajectory to have been generated
            
            Args:
                trajectory: batch of transitions [ (all states), (all actions), (all next states), (all rewards) ]
                
        '''
        
        for _ in range(self.optimizations_per_learn):
            self._learn(trajectory, discount_factor)

        


    
    def interpret_trajectory(self, trajectory):
        
        state_batch = interpret_values(trajectory["state"], self.device)

        action_batch = interpret_values(trajectory["action"], self.device)

        next_state_batch = interpret_values(trajectory["next_state"], self.device)
            
        reward_batch = interpret_unit_values(trajectory["reward"], self.device)

        done_batch = interpret_unit_values(trajectory["done"], self.device)
            
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch
    
    
    def _non_final_states_mask(self, next_state_batch):
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_state_batch)), dtype=torch.bool)

        
        return non_final_mask