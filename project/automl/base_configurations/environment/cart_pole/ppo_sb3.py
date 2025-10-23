



'''
This is based on the stable baselines 3 hyperparameter configuration, in:

https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

With hyperparameters:

# Tuned
CartPole-v1:
  n_envs: 8 # it gets transitions from 8 different environments 
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  n_steps: 32 # in each environment, it computes 32 steps before an update
  batch_size: 256 # it updates after 256 total steps
  gae_lambda: 0.8
  gamma: 0.98
  n_epochs: 20
  ent_coef: 0.0
  learning_rate: lin_0.001
  clip_range: lin_0.2


'''

from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.environment.gymnasium_env import GymnasiumEnvironmentWrapper
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.trainers.agent_trainer_ppo import AgentTrainerPPO
from automl.rl.learners.ppo_learner import PPOLearner



def config_dict():

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": (GymnasiumEnvironmentWrapper, {"environment" : "CartPole-v1"}),

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
            "limit_total_steps" : 1e5,
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