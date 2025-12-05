from automl.component import Component, InputSignature, requires_input_proccess


from automl.rl.agent.agent_components import AgentSchema
import torch

class LearnerSchema(Component):
        
    parameters_signature = {
        "agent" : InputSignature(),
        "optimizations_per_learn" : InputSignature(default_value=1)

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
            self._learn()

    
    def _interpret_trajectory(self, trajectory):
        
        if not isinstance(trajectory.state, torch.Tensor):
            state_batch = torch.stack(trajectory.state, dim=0).to(self.device)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            state_batch = trajectory.state.to(self.device)
            
        
        action_batch = trajectory.action
            
        if not isinstance(trajectory.next_state, torch.Tensor):
            next_state_batch = torch.stack(trajectory.next_state, dim=0).to(self.device)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            next_state_batch = trajectory.next_state.to(self.device)
            
        if not isinstance(trajectory.reward, torch.Tensor):
            
            reward_batch = torch.stack(trajectory.reward, dim=0).to(self.device)  # Stack tensors along a new dimension (dimension 0)
    
        
        else:
            reward_batch = trajectory.reward.view(-1).to(self.device) # TODO: This assumes the reward only has one dimension

        if not isinstance(trajectory.done, torch.Tensor):
            
            done_batch = torch.stack(trajectory.done, dim=0).to(self.device)  # Stack tensors along a new dimension (dimension 0)
    
        
        else:
            done_batch = trajectory.done.view(-1).to(self.device) # TODO: This assumes the reward only has one dimension
            
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch
    
    
    def _non_final_states_mask(self, next_state_batch):
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_state_batch)), dtype=torch.bool)

        
        return non_final_mask