from abc import abstractmethod

from automl.component import ParameterSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentParameterSignature
from automl.utils.shapes_util import double_final_size
import torch

import random

from automl.ml.models.model_components import ModelComponent

from automl.rl.policy.policy import Policy

class StochasticPolicy(Policy):
    '''
    A policy wich selects actions based on distributions
    The model output is used to compute a distribution for the actions
    '''
        
    parameters_signature = {}   
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
        
    def predict(self, state):
        
        model_output = self.predict_model_output(state) # real numbers higher the higher probability        
        
        distribution = self.distribution_from_model_output(model_output) # probabilities computed from logits
        
        return self.predict_from_distribution(distribution)
        
    
    @requires_input_proccess
    def predict_model_output(self, state) -> torch.Tensor:
        return self.model.predict(state)
    
    
    @abstractmethod
    def distribution_from_model_output(self, model_output) -> torch.Tensor:
        pass
    
    
    def predict_from_distribution(self, distribution : torch.distributions):
        return distribution.sample()
    
    def log_probability_of_action(self, distribution, action):
        return distribution.log_prob(action).sum(dim=-1)
    
    
    def predict_from_model_output_with_log(self, model_output):
                
        dist = self.distribution_from_model_output(model_output)
        
        action = self.predict_from_distribution(dist)
                
        log_prob = self.log_probability_of_action(dist, action)

        return action, log_prob    

    
    def predict_with_log(self, state):
        
        model_output = self.predict_model_output(state)    
                
        return self.predict_from_model_output_with_log(model_output)
    


class CategoricalStochasticPolicy(StochasticPolicy):
    '''
    A policy wich selects actions given on categorical probabilities
    '''
        
    parameters_signature = {
    }   
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        
    # EXPOSED METHODS --------------------------------------------------------------------------------------------------------
        
    
    @requires_input_proccess
    def predict_model_output(self, state) -> torch.Tensor:
        '''
        Predicts the logits for the actions
        '''

        return super().predict_model_output(state)
    
    
    def probabilities_from_logits(self, logits) -> torch.Tensor:

        probs = torch.softmax(logits, dim=-1)
        
        return probs

    def distribution_from_model_output(self, model_output) -> torch.Tensor:
        
        probabilities = self.probabilities_from_logits(model_output)
    
        return torch.distributions.Categorical(probs=probabilities)
        

class NormalStochasticPolicy(StochasticPolicy):
    '''
    A policy wich selects actions given on normal distributions for each output
    This means that the shape of the model is mean and standard deviation for each action value it can compute
    '''

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def _compute_model_output_shape(self):
        
        self.lg.writeLine(f"Action output shape is {self.output_action_shape}")
        # model outputs mean and std for each dimension
        self.model_output_shape = double_final_size(self.output_action_shape)

        self.lg.writeLine(f"Normal policy model output shape is {self.model_output_shape}")


    def distribution_from_model_output(self, model_output):

        '''
        Converts model output into a Normal distribution.
        Model output is assumed to contain:
            [mean_1 ... mean_A , log_std_1 ... log_std_A]
        '''

        action_dim = model_output.shape[-1] // 2

        mean = model_output[..., :action_dim]
        log_std = model_output[..., action_dim:]

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # convert log_std → std
        std = torch.exp(log_std)

        return torch.distributions.Normal(mean, std)
    


class ConstrainedNormalStochasticPolicy(NormalStochasticPolicy):
    '''
    A policy wich selects actions given on normal distributions for each output
    This means that the shape of the model is mean and standard deviation for each action value it can compute

    It passes through a tanh activation function
    '''

    EPS = 1e-6

    def _compute_model_output_shape(self):
        
        super()._compute_model_output_shape()

        self._get_bounds_from_shape()

        self.action_range = self.max_action_value - self.min_action_value

        self.lg.writeLine(f"Bound of actions for policies is [{self.min_action_value}, {self.max_action_value}] with a range of {self.action_range}")

    
    def _get_bounds_from_shape(self):

        if hasattr(self.output_action_space, "low"):
            self.lg.writeLine(f"Output shape has lower bound: {self.output_action_space.low}")

        if hasattr(self.output_action_space, "high"):
            self.lg.writeLine(f"Output shape has lower bound: {self.output_action_space.high}")

        # gym-style Box bounds
        self.min_action_value = torch.as_tensor(
            self.output_action_space.low
        )

        self.max_action_value = torch.as_tensor(
            self.output_action_space.high
        )


    
    def predict_from_distribution(self, distribution : torch.distributions):
       return self.min_action_value + (self.action_range) * 0.5 * (torch.tanh(distribution.sample()) + 1.0) 


    def log_probability_of_action(self, distribution, action):
        low = self.min_action_value.to(action.device)
        high = self.max_action_value.to(action.device)

        # map action from [low, high] -> [-1, 1]
        y = 2.0 * (action - low) / (high - low) - 1.0
        y = torch.clamp(y, -1.0 + self.EPS, 1.0 - self.EPS)

        # inverse tanh
        raw_action = 0.5 * torch.log((1.0 + y) / (1.0 - y))

        # log prob in raw Gaussian space
        log_prob = distribution.log_prob(raw_action)

        # jacobian correction
        log_det_jacobian = torch.log((high - low) / 2.0) + torch.log(1.0 - y.pow(2) + self.EPS)

        return (log_prob - log_det_jacobian).sum(dim=-1)