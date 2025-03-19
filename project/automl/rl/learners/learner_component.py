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

