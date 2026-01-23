from types import FunctionType
from automl.component import Component, InputSignature, requires_input_proccess
from automl.core.advanced_input_management import ComponentListInputSignature
from automl.basic_components.sampler import Sampler
import torch
import random
import math
import numpy as nn


class EnvironmentComponent(Component):

    '''Represents any environment'''
    
    
    @requires_input_proccess
    def close(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_env_name(self):
        raise NotImplementedError()

    @requires_input_proccess    
    def reset(self):
        '''A soft reset of the environment, only to guarantee it is in its initial state'''
        raise NotImplementedError()
    
    @requires_input_proccess    
    def total_reset(self):
        '''Resets all, including RNG state'''
        raise NotImplementedError()
    



    
def sampled_environment_fun(func : FunctionType):

    '''
    Returns a function that samples an environment if there is none yet and calls the passed function from the environment
    '''
        
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
        
        self.environment_input = self.get_input_value("environment_input")
                
    def sample(self) -> EnvironmentComponent:
        raise NotImplementedError()

    @sampled_environment_fun
    def close(self):
        raise NotImplementedError()
    
    @sampled_environment_fun
    def get_env_name(self):
        raise NotImplementedError()
    
    
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

        self.environments : EnvironmentComponent = self.get_input_value("environments") 
        self.generate_name = self.get_input_value("generate_name")
        self.next_index = 0

        if self.generate_name:
            for index in range(len(self.environments)):

                env = self.environments[index]
                generated_name = f"{env.name}_{index}"
                env.pass_input({"name" : generated_name})


    
    @requires_input_proccess
    def sample(self) -> EnvironmentComponent:

        to_return = self.environments[self.next_index]
        self.next_index = ( self.next_index + 1 ) % len(self.environments)

        return to_return


    
     
    

     