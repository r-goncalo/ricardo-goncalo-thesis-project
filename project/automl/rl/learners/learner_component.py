from automl.component import Component, InputSignature, requires_input_proccess

from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer


import torch

class LearnerSchema(Component):
        
    parameters_signature = {
        "agent" : InputSignature(),

    }
        
    def learn(self, trajectory, discount_factor) -> None:
        
        '''
            Learns the policy using the current policy and the trajectory it is supposed to learn
            
            Remember that On-Policy algorithms expect the trajectory to have been generated
            
            Args:
                trajectory: batch of transitions [ (all states), (all actions), (all next states), (all rewards) ]
                
        '''
        
        pass

    
    def _interpret_trajectory(self, trajectory):
        
        if not isinstance(trajectory.state, torch.Tensor):
            state_batch = torch.stack(trajectory.state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            state_batch = trajectory.state
            
        
        action_batch = trajectory.action
            
        if not isinstance(trajectory.next_state, torch.Tensor):
            next_state_batch = torch.stack(trajectory.next_state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            next_state_batch = trajectory.next_state
            
        if not isinstance(trajectory.reward, torch.Tensor):
            
            reward_batch = torch.stack(trajectory.reward, dim=0)  # Stack tensors along a new dimension (dimension 0)
    
        
        else:
            reward_batch = trajectory.reward.view(-1) # TODO: This assumes the reward only has one dimension
            
        return state_batch, action_batch, next_state_batch, reward_batch
    
    
    def _non_final_states_mask(self, next_state_batch):
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_state_batch)), dtype=torch.bool)

        
        return non_final_mask