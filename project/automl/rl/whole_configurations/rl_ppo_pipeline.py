from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.policy.qpolicy import QPolicy
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.learners.ppo_learner import PPOLearner

from automl.rl.environment.pettingzoo.pettingzoo_env import PettingZooEnvironmentWrapper
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.rl.trainers.agent_trainer_ppo import AgentTrainerPPO
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent


def config_dict(num_episodes=200):

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": {
            "__type__": PettingZooEnvironmentWrapper,
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "state_memory_size" : 2,

            "policy" : ( StochasticPolicy,
                        {
                        "model" : (
                            FullyConnectedModelSchema, 
                            {
                            "hidden_layers" : 3,
                            "hidden_size" : 64
                            }
                            ),
                        }
                )
        },
        
        "rl_trainer" : (RLTrainerComponent,
            
            {
            "default_trainer_class" : AgentTrainerPPO,
            "num_episodes" : num_episodes,
            "optimization_interval": 300,
            "agents_trainers_input" : { #for each agent trainer
                
                "learner" : (PPOLearner, {}),
            
                "memory" : (TorchDiskMemoryComponent, {
                    "capacity" : 500
                })
                
            
            }

            }
        )
        
    },
    "child_components": [
        {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader",
            "input": {
                "environment": "cooperative_pong"
            }
        }
    ]
}


