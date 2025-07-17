import itertools
from automl.basic_components.seeded_component import SeededComponent
from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, InputSignature, requires_input_proccess
from automl.rl.environment.environment_components import EnvironmentComponent


from automl.utils.shapes_util import torch_state_shape_from_space

import gymnasium as gym
import torch


class GymnasiumEnvironmentWrapper(EnvironmentComponent, SeededComponent, StatefulComponent):
    
    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "environment": InputSignature(default_value="CartPole-v1"),
        "render_mode": InputSignature(default_value="rgb_array", validity_verificator=lambda x: x in ["rgb_array", "human"]),
        "device": InputSignature(ignore_at_serialization=True)
    }

    @staticmethod
    def state_translator(state, device):
                
        with torch.no_grad():
            
            return torch.tensor(state, dtype=torch.float32, device=device)


    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.device = self.input["device"]
        self.setup_environment()
        self.reset()
        
        self.last_observation = None
        self.last_reward = 0
        self.last_done = False
        self.last_info = {}
        
        self.reset_info = {}
        

    def setup_environment(self):
        
        if isinstance(self.input["environment"], str):
            self.load_environment(self.input["environment"])
        
        elif isinstance(self.input["environment"], gym.Env):
            self.env: gym.Env = self.input["environment"]
        
        else:
            raise Exception("No valid Gymnasium environment or environment name passed.")


    def load_environment(self, environment_name: str):
        
        try:
            self.env: gym.Env = gym.make(environment_name, render_mode=self.input["render_mode"])
        
        except Exception as e:
            raise Exception(f"{self.name}: Failed to load gym environment '{environment_name}': {str(e)}")


    @requires_input_proccess
    def get_agent_action_space(self, agent):
        '''returns the action space for the given agent'''
        return self.env.action_space
    
    @requires_input_proccess
    def get_agent_state_space(self, agent):
        '''returns the state space for the environment'''
                
        internal_state_shape = torch_state_shape_from_space(self.env.observation_space)
                
        return internal_state_shape

    
    def reset(self):
        
        # TODO: Check if this is well done (in regards to seeds, should it not be the same)
        observation, info = self.env.reset()
        
        self.reset_info = info
        
        #observation, _ = self.env.reset(seed=self._seed)
        self.last_observation = observation
    
        return self.state_translator(observation, self.device)
    

    def last(self):
        return self.state_translator(self.last_observation, self.device), self.last_reward, self.last_done, self.last_info


    def agents(self):
        return ["agent"]


    def agent_iter(self):
        return itertools.repeat("agent")


    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)        
        
        done = terminated or truncated
        
        self.last_observation = obs
        self.last_reward = reward
        self.last_done = done
        self.last_info = info
                
        return self.state_translator(obs, self.device), reward, done, info


    def render(self):
        return self.env.render()


    def close(self):
        return self.env.close()


    def observation_space(self):
        return self.env.observation_space

    
    def observe(self, *args):
        return self.state_translator(self.last_observation, self.device)
    
    def get_env_info(self):
        return self.reset_info
    

    # STATE MANAGEMENT -------------------------------------------------------------------
    
    def on_unload(self):
        
        super().on_unload()
        
        self.env.close()
