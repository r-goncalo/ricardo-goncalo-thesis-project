from automl.basic_components.seeded_component import SeededComponent
from automl.component import Component, InputSignature, requires_input_proccess

from automl.fundamentals.translator.translator import Translator
from automl.rl.environment.parallel_environment import ParallelEnvironmentComponent
import torch

from pettingzoo import ParallelEnv



class PettingZooEnvironmentWrapperParallel(ParallelEnvironmentComponent, SeededComponent):

    parameters_signature = { 
        "environment": InputSignature(default_value="cooperative_pong"),
        "render_mode": InputSignature(default_value="rgb_array", validity_verificator=lambda x: x in ["rgb_array", "human"]),
        "device": InputSignature(ignore_at_serialization=True)
    }


    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.render_mode = self.get_input_value("render_mode")
        self.device = self.get_input_value("device")

        self._setup_environment()


    def _setup_environment(self):
        env_input = self.get_input_value("environment")

        if isinstance(env_input, str):
            self._load_environment(env_input)

        elif isinstance(env_input, ParallelEnv):
            self.env: ParallelEnv = env_input

        else:
            raise Exception("Invalid environment for PettingZooEnvironmentWrapper")


    def _load_environment(self, environment_name: str):

        if environment_name == "cooperative_pong":
            from pettingzoo.butterfly import cooperative_pong_v5
            self.env: ParallelEnv = cooperative_pong_v5.parallel_env(render_mode=self.render_mode)

        else:
            raise Exception(f"No valid PettingZoo environment named '{environment_name}'")
        
        self.env_name = environment_name


    def observe(self, agent):
        """
        Parallel API: observations are not retrieved with env.observe(agent)
        but from the dictionary returned by reset() and step().
        This method simply accesses the last stored obs dict.
        """
        return self._last_obs.get(agent, None)


    @requires_input_proccess
    def get_agent_action_space(self, agent):
        return self.env.action_space(agent)


    @requires_input_proccess
    def get_agent_state_space(self, agent):
        return self.env.observation_space(agent)


    @requires_input_proccess
    def agents(self):

        to_return = None
        
        if hasattr(self.env, "agents"):
            to_return = list(self.env.agents)


        if (to_return is None or len(to_return) == 0) and hasattr(self.env, "possible_agents"):
            to_return = list(self.env.possible_agents)

        if to_return is None:
            raise AttributeError("Environment has neither .agents nor .possible_agents")
        else:
            return to_return
        
    
    def parallel_agents(self):
        return self.env.agents

    def reset(self):
        """
        Returns:
            obs_dict[agent] = observation
            info_dict[agent] = info
        """
        obs, info = self.env.reset()
        self._last_obs = obs  # store so .observe(agent) can work
        self.reset_info = info
        return obs
    
    def total_reset(self):
        """
        Returns:
            obs_dict[agent] = observation
            info_dict[agent] = info
        """
        obs, info = self.env.reset(seed=self.seed)
        self._last_obs = obs  # store so .observe(agent) can work
        self.reset_info = info
        return obs


    def step(self, actions):
        """
        Args:
            actions: {agent: action}
        Returns:
            next_obs_dict, rewards, terminations, truncations, infos
        """
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self._last_obs = next_obs
        return next_obs, rewards, terminations, truncations, infos


    # For debugging or compatibility
    def rewards(self):
        return self.env.rewards

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @requires_input_proccess
    def get_env_name(self):
        return self.env_name
