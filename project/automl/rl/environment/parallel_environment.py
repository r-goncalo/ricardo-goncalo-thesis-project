
from automl.rl.environment.environment_components import EnvironmentComponent
from automl.component import requires_input_proccess


class ParallelEnvironmentComponent(EnvironmentComponent):
    
    parameters_signature =  {} 
        
    @requires_input_proccess    
    def reset(self):
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