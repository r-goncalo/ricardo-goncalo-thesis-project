from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.policy.qpolicy import QPolicy

from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent

def config_dict(num_episodes=200):

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "optimization_interval": 100,
        "device" : "cuda",
        "environment": {
            "__type__": PettingZooEnvironmentWrapper,
            "name": "PettingZooEnvironmentLoader"        
        },
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
            "num_episodes" : num_episodes,
            "optimization_interval": 300,
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
    
def mockup_config_dict(num_episodes=200):
    
    from automl.ml.models.mockups.model_mockup import MockupRandomModel
    from automl.ml.optimizers.mockups.mockup_optimizers import MockupOptimizerSchema

    return {
    
    "__type__": "<class 'automl.rl.rl_pipeline.RLPipelineComponent'>",
    "name": "RLPipelineComponent",
    "input": {
        "num_episodes_per_run": num_episodes,
        "optimization_interval": 300,
        "device" : "cpu",
        "environment": {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "exploration_strategy_class" : str(EpsilonGreedyStrategy),
            "state_memory_size" : 2,
            "exploration_strategy_input" : {
                "epsilon_end" : 0.1,
                "epsilon_start" : 0.99,
                "epsilon_decay" : 0.99
                },
            "policy" : ( QPolicy,
                        {
                        "model" : ( MockupRandomModel,{})
                        }
                ),
            "learner" : (DeepQLearnerSchema, {
                "target_update_rate" : 0.05,
                "optimizer" :( MockupOptimizerSchema, {})
            }),
            "memory_input" : {
                "capacity" : 300
            }
        },

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