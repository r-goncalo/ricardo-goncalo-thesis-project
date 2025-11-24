from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.policy.qpolicy import QPolicy

from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.agent_trainer_component_dqn import AgentTrainerDQN
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent


def config_dict():


    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": (PettingZooEnvironmentWrapper, {
                            "environment": "cooperative_pong"}),
        "agents_input": {
            "state_memory_size" : 2,

            "policy" : ( QPolicy,
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
            "num_episodes" : 100,
            "optimization_interval": 300,
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "learner" : (DeepQLearnerSchema, {
                               "target_update_rate" : 0.05,
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 0.0001
                                   }
                )
                }),
            
                "memory" : (TorchDiskMemoryComponent, {
                    "capacity" : 500
                }),
                
                "exploration_strategy" : (EpsilonGreedyStrategy,
                                                                  {
                                            "epsilon_end" : 0.1,
                                            "epsilon_start" : 0.99,
                                            "epsilon_decay" : 0.99
                                                                  }
                                          )
    
                    }

            }
        )
        
    }
}