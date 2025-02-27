from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema

def config_dict():

    return {
    
    "__type__": "<class 'automl.rl.rl_pipeline.RLPipelineComponent'>",
    "name": "RLPipelineComponent",
    "input": {
        "num_episodes": 3,
        "state_memory_size": 2,
        "limit_steps": 200,
        "optimization_interval": 50,
        "device" : "cuda",
        "environment": {
            "__type__": "<class 'automl.rl.environment.environment_components.PettingZooEnvironmentLoader'>",
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "exploration_strategy_class" : str(EpsilonGreedyStrategy),
            "model_class" : str(FullyConnectedModelSchema),
            "model_input" : {
                "hidden_layers" : 3,
                "hidden_size" : 64
            }
        },
        "save_interval": 100
    },
    "child_components": [
        {
            "__type__": "<class 'automl.rl.environment.environment_components.PettingZooEnvironmentLoader'>",
            "name": "PettingZooEnvironmentLoader",
            "input": {
                "petting_zoo_environment": "cooperative_pong"
            }
        }
    ]
}