from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema

import torch

from automl.rl.policy.policy import Policy


class DeepQLearnerSchema(LearnerSchema):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "target_update_rate" : InputSignature(default_value=0.05),
                        "update_target_at_optimization" : InputSignature(default_value=True),
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                        "optimizer" : ComponentInputSignature(
                            default_component_definition=(
                                AdamOptimizer,
                                {}
                            )
                            )

                        }    
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.agent : Component = self.input["agent"]
        
        self.device = self.input["device"]
        
        self.TAU = self.input["target_update_rate"] #the update rate of the target network
        
        self.update_target_at_optimization = self.input["update_target_at_optimization"]
        
        self.policy : Policy = self.agent.get_policy()
        
        self.model = self.policy.model
        
        self.target_net = self.model.clone() #the target network has the same initial parameters as the policy being trained

        self.initialize_optimizer()
        
        
    def initialize_optimizer(self):
        
        self.optimizer = ComponentInputSignature.get_component_from_input(self, "optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    @requires_input_proccess
    def learn(self, trajectory, discount_factor) -> None:
        
        super().learn(trajectory, discount_factor)
        
        if not isinstance(trajectory.state, torch.Tensor):
            state_batch = torch.stack(trajectory.state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            state_batch = trajectory.state
            
        if not isinstance(trajectory.next_state, torch.Tensor):
            next_state_batch = torch.stack(trajectory.next_state, dim=0)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            next_state_batch = trajectory.next_state
            
        if not isinstance(trajectory.reward, torch.Tensor):
            
            reward_batch = torch.stack(trajectory.reward, dim=0)  # Stack tensors along a new dimension (dimension 0)
    
        
        else:
            reward_batch = trajectory.reward.view(-1) # TODO: This assumes the reward only has one dimension
            
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_state_batch)), dtype=torch.bool)
        
        non_final_next_states = torch.stack([s for s in next_state_batch
                                                        if s is not None], dim=0)
                
        
        #predict the action we would take given the current state
        predicted_actions_values = self.model.predict(state_batch)
        predicted_actions_values, predicted_actions = predicted_actions_values.max(1)
        
        #compute the q values our target net predicts for the next_state (perceived reward)
        #if there is no next_state, we can use 0
                
        next_state_values = torch.zeros(len(trajectory[0]), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.predict(non_final_next_states).max(1).values # it returns the maximum q-action values of the next action
            
        # Compute the expected Q values (the current reward of this state and the perceived reward we would get in the future)
        next_state_values.mul_(discount_factor).add_(reward_batch)
                
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_actions_values, next_state_values)        
        
        if self.update_target_at_optimization:
            self.update_target_model()
        
        
    @requires_input_proccess            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.model, self.TAU)