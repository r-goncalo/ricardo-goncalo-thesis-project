import itertools
from types import FunctionType
from automl.basic_components.sampler import Sampler
from automl.basic_components.seeded_component import SeededComponent
from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, InputSignature, requires_input_proccess
from automl.rl.environment.environment_components import EnvironmentComponent


from automl.utils.shapes_util import torch_state_shape_from_space

import gymnasium as gym
from automl.core.advanced_input_management import ComponentListInputSignature
import torch


def sampled_environment_fun(func : FunctionType):
        
    def wrapper(self : Sampler, *args, **kwargs):
        
        if self.sampled_environment == None: # samples environment if non existent
            self.sampled_environment = self.sample()

        func(self, *args, **kwargs) # evaluates the function on the sampler, probabily not needed
        return getattr(self.sampled_environment, func.__name__)(*args, **kwargs) # the function is actually evaluated in the environment

    return wrapper

class EnvironmentSampler(Sampler, EnvironmentComponent):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environment_input": InputSignature(default_value={}),
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        self.environment_input = self.input["environment_input"]
                
        
    def sample(self) -> EnvironmentComponent:
        raise NotImplementedError()
        
        

    @sampled_environment_fun    
    def reset(self):
        pass
            
    @sampled_environment_fun   
    def observe(self, *args):
        pass
        
    @sampled_environment_fun   
    def agents(self):
        pass
    
    @sampled_environment_fun
    def get_agent_action_space(self, agent):
        pass
    
    @sampled_environment_fun
    def get_agent_state_space(self, agent):
        pass

    
    @sampled_environment_fun   
    def last(self):
        pass
    
    @sampled_environment_fun   
    def agent_iter(self):
        pass
    
    @sampled_environment_fun    
    def step(self, *args):
        pass
        
    @sampled_environment_fun    
    def rewards(self):
        pass
    
    @sampled_environment_fun
    def render(self):
        pass
    
    @sampled_environment_fun
    def close(self):
        pass
    
    @sampled_environment_fun
    def get_env_name(self):
        pass
    
class EnvironmentCycler(EnvironmentSampler):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environments": ComponentListInputSignature(),
        "generate_name" : InputSignature(default_value=False)
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.environments : EnvironmentComponent = ComponentListInputSignature.get_component_list_from_input(self, "environments") 
        self.generate_name = self.input["generate_name"]
        self.next_index = 0

        if self.generate_name:
            for index in range(len(self.environments)):
                env = self.environments[index]
                env.pass_input({"name" : f"{env.name}_{index}"})

    
    @requires_input_proccess
    def sample(self) -> EnvironmentComponent:

        to_return = self.environments[self.next_index]
        self.next_index = ( self.next_index + 1 ) % len(self.environments)

        return to_return
