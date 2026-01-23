

from automl.rl.environment.environment_components import EnvironmentComponent, EnvironmentSampler, sampled_environment_fun
from automl.component import requires_input_proccess


class AECEnvironmentComponent(EnvironmentComponent):
    
    parameters_signature =  {} 
    
        
    @requires_input_proccess    
    def reset(self):
        '''A soft reset of the environment, only to guarantee it is in its initial state'''
        raise NotImplementedError()
    
    @requires_input_proccess    
    def total_reset(self):
        '''Resets all, including RNG state'''
        raise NotImplementedError()
        
    @requires_input_proccess   
    def observe(self, *args):
        raise NotImplementedError()
        
    @requires_input_proccess   
    def agents(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_agent_state_space(self, agent):
        '''returns the state space for the environment'''
        raise NotImplementedError()
    
    @requires_input_proccess   
    def last(self):
        raise NotImplementedError()
    
    @requires_input_proccess   
    def agent_iter(self):
        raise NotImplementedError()
    
    @requires_input_proccess    
    def step(self, *args):
        raise NotImplementedError()
        
    @requires_input_proccess    
    def rewards(self):
        raise NotImplementedError()    
    
    @requires_input_proccess
    def render(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def close(self):
        raise NotImplementedError()
    
    @requires_input_proccess
    def get_env_name(self):
        raise NotImplementedError()
    

class AECEnvironmentSampler(EnvironmentSampler, AECEnvironmentComponent):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
    }

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.sampled_environment = None

    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        self.environment_input = self.get_input_value("environment_input")
                
        
    def sample(self) -> AECEnvironmentComponent:
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
    
class AECEnvironmentCycler(AECEnvironmentSampler):
    
    '''
    A component which samples environments and also works as a wrapper for them
    This means it can be used as an environment it is able to sample
    
    '''
