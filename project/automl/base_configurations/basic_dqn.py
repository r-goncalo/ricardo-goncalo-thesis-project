from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.policy.qpolicy import QPolicy

from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper

def config_dict(num_episodes=200):

    return {
    
    "__type__": "<class 'automl.rl.rl_pipeline.RLPipelineComponent'>",
    "name": "RLPipelineComponent",
    "input": {
        "num_episodes_per_run": num_episodes,
        "state_memory_size": 2,
        "limit_steps": 1000,
        "optimization_interval": 100,
        "device" : "cuda",
        "environment": {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "exploration_strategy_class" : str(EpsilonGreedyStrategy),
            "exploration_strategy_input" : {
                "epsilon_end" : 0.1,
                "epsilon_start" : 0.99,
                "epsilon_decay" : 0.99
                },
            "policy_class" : str(QPolicy),
            "policy_input" : {
                "model" : (
                    FullyConnectedModelSchema, 
                    {
                    "hidden_layers" : 3,
                    "hidden_size" : 64
                    }
                    ),
            },
            "learner_input" : {
                "target_update_rate" : 0.05,
                "optimizer" :(
                    AdamOptimizer,
                    {
                        "learning_rate" : 0.0001
                    }
                )

            },
            "memory_input" : {
                "capacity" : 300
            }
        },
        "save_interval": 100
    },
    "child_components": [
        {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader",
            "input": {
                "petting_zoo_environment": "cooperative_pong"
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
        "state_memory_size": 2,
        "limit_steps": 50,
        "optimization_interval": 300,
        "device" : "cpu",
        "environment": {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "exploration_strategy_class" : str(EpsilonGreedyStrategy),
            "exploration_strategy_input" : {
                "epsilon_end" : 0.1,
                "epsilon_start" : 0.99,
                "epsilon_decay" : 0.99
                },
            "policy_class" : str(QPolicy),
            "policy_input" : {
                "model" : (MockupRandomModel, {}),
            },
            "learner_input" : {
                "target_update_rate" : 0.05,
                "optimizer" :( MockupOptimizerSchema, {})
            },
            "memory_input" : {
                "capacity" : 300
            }
        },
        "save_interval": 100
    },
    "child_components": [
        {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader",
            "input": {
                "petting_zoo_environment": "cooperative_pong"
            }
        }
    ]
}