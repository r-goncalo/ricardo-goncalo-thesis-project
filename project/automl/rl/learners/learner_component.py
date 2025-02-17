from automl.component import Schema, InputSignature, requires_input_proccess

from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer

from abc import abstractmethod

import torch

class LearnerSchema(Schema):
        
    @abstractmethod
    def learn(self, trajectory) -> None:
        
        '''
            Learns the policy using the current policy and the trajectory it is supposed to learn
            
            Remember that On-Policy algorithms expect the trajectory to have been generated
            
            Args:
                trajectory: batch of transitions [ (all states), (all actions), (all next states), (all rewards) ]
                
        '''
        
        pass

class DeepQLearnerSchema(LearnerSchema):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        
                       "agent" : InputSignature(),
                       "target_update_rate" : InputSignature(default_value=0.05),
                        "update_target_at_optimization" : InputSignature(default_value=True),
                        "device" : InputSignature(ignore_at_serialization=True),
                        "optimizer" : InputSignature(generator= lambda self : self.initialize_child_component(AdamOptimizer), possible_types=[OptimizerSchema]),

                        }    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.agent = self.input["agent"]
        
        self.device = self.input["device"]
        
        self.TAU = self.input["target_update_rate"] #the update rate of the target network
        
        self.update_target_at_optimization = self.input["update_target_at_optimization"]
        
        self.policy = self.agent.get_policy()
        
        self.target_net = self.policy.clone() #the target network has the same initial parameters as the policy being trained

        self.initialize_optimizer()
        
        
    def initialize_optimizer(self):
        self.optimizer : OptimizerSchema = self.input["optimizer"]
        self.optimizer.pass_input({"model_params" : self.policy.get_model_params()})

    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    @requires_input_proccess
    def learn(self, trajectory, discount_factor) -> None:
        
        super().learn(trajectory)
                
        state_batch = torch.stack(trajectory.state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        reward_batch = torch.tensor(trajectory.reward, device=self.device)
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              trajectory.next_state)), dtype=torch.bool)
        
        non_final_next_states = torch.stack([s for s in trajectory.next_state
                                                        if s is not None], dim=0)
                
        
        #predict the action we would take given the current state
        predicted_actions_values = self.policy.predict(state_batch)
        predicted_actions_values, predicted_actions = predicted_actions_values.max(1)
        
        #compute the q values our target net predicts for the next_state (perceived reward)
        #if there is no next_state, we can use 0
                
        next_state_values = torch.zeros(len(trajectory[0]), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.predict(non_final_next_states).max(1).values
            
        # Compute the expected Q values (the current reward of this state and the perceived reward we would get in the future)
        expected_state_action_values = (next_state_values * discount_factor) + reward_batch
                
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_actions_values, expected_state_action_values)        
        
        if self.update_target_at_optimization:
            self.update_target_model()
        
        
    @requires_input_proccess            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.policy, self.TAU)