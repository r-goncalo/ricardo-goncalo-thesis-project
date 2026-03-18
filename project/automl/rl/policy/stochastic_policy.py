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
        
        action_val = self.sample_action_val_from_distribution(distribution)

        return self.get_action_from_action_val(action_val)
    
    @requires_input_proccess
    def get_action_val_shape(self):
        return self.output_action_shape

    @requires_input_proccess 
    def get_action_from_action_val(self, action_val):
        return action_val
    
    @requires_input_proccess
    def predict_model_output(self, state) -> torch.Tensor:
        return self.model.predict(state)
    
    
    @abstractmethod
    def distribution_from_model_output(self, model_output) -> torch.Tensor:
        pass
    
    
    def sample_action_val_from_distribution(self, distribution : torch.distributions):
        return distribution.sample()
    
    def log_probability_of_action_val(self, distribution, action_val):
        return distribution.log_prob(action_val).sum(dim=-1)
    
    
    def predict_action_val_from_model_output_with_log(self, model_output):
                
        dist = self.distribution_from_model_output(model_output)
        
        action_val = self.sample_action_val_from_distribution(dist)
                
        log_prob = self.log_probability_of_action_val(dist, action_val)

        return action_val, log_prob    

    
    def predict_action_val_with_log(self, state):
        
        model_output = self.predict_model_output(state)    
                
        return self.predict_action_val_from_model_output_with_log(model_output)
    


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

    def log_probability_of_action_val(self, distribution, action_val):
        return distribution.log_prob(action_val)
    
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

        if hasattr(self.output_action_shape, "low"):
            self.lg.writeLine(f"Output shape has lower bound: {self.output_action_shape.low}")

        if hasattr(self.output_action_shape, "high"):
            self.lg.writeLine(f"Output shape has lower bound: {self.output_action_shape.high}")

        # gym-style Box bounds
        self.min_action_value = torch.as_tensor(
            self.output_action_shape.low,
            device=self.device
        )

        self.max_action_value = torch.as_tensor(
            self.output_action_shape.high,
            device=self.device
        )
    
    def get_action_from_action_val(self, action_val):
        return self.min_action_value + (self.action_range) * 0.5 * (torch.tanh(action_val) + 1.0) 
