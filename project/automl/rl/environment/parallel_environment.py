
from automl.rl.environment.environment_components import EnvironmentComponent
from automl.component import requires_input_proccess

from abc import abstractmethod

class ParallelEnvironmentComponent(EnvironmentComponent):
    
    parameters_signature =  {} 
        

    @abstractmethod
    def step(self, actions):
        """
        Args:
            actions: {agent: action}
        Returns:
            next_obs_dict, rewards, terminations, truncations, infos
        """
        pass

        