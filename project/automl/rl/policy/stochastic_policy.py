from automl.component import InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
import torch

import random

from automl.ml.models.model_components import ModelComponent



from automl.rl.policy.policy import Policy

class StochasticPolicy(Policy):
    '''
    A policy wich selects actions given on probabilities
    '''
        
    parameters_signature = {
    }   
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    def predict(self, state):
        
        logits = self.predict_logits(state) # real numbers higher the higher probability        
        
        probs = self.probabilities_from_logits(logits) # probabilities computed from logits
        
        return self.predict_from_probability(probs)
        
    
    @requires_input_proccess
    def predict_logits(self, state) -> torch.Tensor:
        
        probabilitiesForActionsLogits : torch.Tensor = self.model.predict(state)

                
        return probabilitiesForActionsLogits
    
    
    def probabilities_from_logits(self, logits) -> torch.Tensor:

        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    
    def predict_from_probability(self, probs):
    
        dist = torch.distributions.Categorical(probs=probs)
        
        sampled_action = dist.sample()
                
        return sampled_action
    
    
    
    def predict_from_probability_with_log(self, probs):
                
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
                
        log_prob = dist.log_prob(action)
        return action, log_prob    

    
    def predict_with_log(self, state):
        
        logits = self.predict_logits(state) # real numbers higher the higher probability        
        
        probs = self.probabilities_from_logits(logits) # probabilities computed from logits
        
        return self.predict_from_probability_with_log(probs)
        
    
    @requires_input_proccess
    def random_prediction(self):
        return torch.randint(
                0, #low
                self.model_output_shape.n, #high
                (1,) #size
            )