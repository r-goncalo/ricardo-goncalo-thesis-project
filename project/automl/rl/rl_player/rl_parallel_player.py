from automl.rl.rl_player.rl_player import RLPlayer
import torch
from automl.rl.environment.parallel_environment import ParallelEnvironmentComponent


class RLParallelPlayer(RLPlayer):

    def _proccess_input_internal(self):
        """
        Extend RLPlayer's input processing, but ensure environment is ParallelEnvironmentComponent.
        """
        super()._proccess_input_internal()

        self.env : ParallelEnvironmentComponent = self.env

        if not isinstance(self.env, ParallelEnvironmentComponent):
            raise Exception(f"RLParallelPlayer requires ParallelEnvironmentComponent, got {type(self.env)}")



    def _do_agent_step_parallel(self):

        actions = {}

        with torch.no_grad():
            for agent_name, agent in self.agents.items():

                act = agent.policy_predict_with_memory()
                actions[agent_name] = act.item()

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        done = False
        total_reward = 0.0

        for agent_name in self.agents.keys():
            total_reward += rewards[agent_name]
            self.agents[agent_name].update_state_memory(observations[agent.name])

            if terminations[agent_name] or truncations[agent_name]:
                done = True

        return total_reward, done



    def _run_episode(self):
        """
        Overrides RLPlayer.__run_episode()  
        Uses a single parallel step per loop rather than iterating over agents.
        """

        while True:

            reward, done = self._do_agent_step_parallel()

            self.values["episode_score"] += reward
            self.values["episode_steps"] += 1
            self.values["total_steps"] += 1

            if done:
                break

            if self.limit_steps > 0 and self.values["episode_steps"] >= self.limit_steps:
                break
