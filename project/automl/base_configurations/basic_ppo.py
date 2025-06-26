from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.learners.ppo_learner import PPOLearner

from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper

def config_dict(num_episodes=200):

    raise NotImplementedError("Basic PPO configuration is not implemented yet.")