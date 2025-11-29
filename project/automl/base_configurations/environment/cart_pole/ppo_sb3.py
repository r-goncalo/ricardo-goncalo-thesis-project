



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

from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.environment.gymnasium.gymnasium_env import GymnasiumEnvironmentWrapper
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.trainers.agent_trainer_ppo import AgentTrainerPPO
from automl.rl.learners.ppo_learner import PPOLearner
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent

from automl.ml.memory.torch_memory_component import TorchMemoryComponent

def config_dict():

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": (GymnasiumEnvironmentWrapper, {"environment" : "CartPole-v1"}),

        "agents_input": {

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
            "name" : "RLTrainerComponent",
            "default_trainer_class" : AgentTrainerPPO,
            "limit_total_steps" : 1e5,
            "predict_optimizations_to_do" : True,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval": 256,
                
                "learner" : (PPOLearner, {

                    "lamda_gae" : 0.8,

                    "critic_model" : (FullyConnectedModelSchema, {"hidden_layers" : 2, "hidden_size" : 64, "output_shape" : 1}),

                    "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "name" : "AdamOpimizerComponent",
                                       "learning_rate" : 2.3e-3,
                                       "linear_decay_learning_rate_with_final_input_value_of" : ("relative", [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]),
                                       "clip_grad_value" : (
                                           DynamicLinearValueInRangeBasedOnComponent, {
                                               "input_for_fun_key" : "optimizations_done",
                                               "initial_value" : 0.2,
                                               "final_value" : 0,
                                               "input_component" : ('relative', ("__get_by_name__", {"name_of_component" : "AdamOpimizerComponent"})),
                                               "input_for_fun_max_value" : 
                                                ('relative', 
                                                 [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]
                                                )


                                           }
                                       )                 
                                   }
                    ),



                }),

                "discount_factor" : 0.98,
            
                "times_to_learn" : 20,
            
                "memory" : (TorchMemoryComponent, {
                    "capacity" : 256
                })
                
            
            }

            }
        )
        
    }
}



def agent_and_agent_trainer():
    
    agent =    {
        
    "__type__": RLPipelineComponent,
    "name": "agent",
    "input": {

    
        
        }
    }

    agent_trainer = {
        
    "__type__": AgentTrainerPPO,
    "name": "agent_trainer",

    "input": { #for each agent trainer
                
                "optimization_interval": 256,
                
                "learner" : (PPOLearner, {

                    "lamda_gae" : 0.8,

                    "critic_model" : (FullyConnectedModelSchema, {"hidden_layers" : 2, "hidden_size" : 64, "output_shape" : 1}),

                    "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "name" : "AdamOpimizerComponent",
                                       "learning_rate" : 2.3e-3,
                                       "linear_decay_learning_rate_with_final_input_value_of" : ("relative", [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]),
                                       "clip_grad_value" : (
                                           DynamicLinearValueInRangeBasedOnComponent, {
                                               "input_for_fun_key" : "optimizations_done",
                                               "initial_value" : 0.2,
                                               "final_value" : 0,
                                               "input_component" : ('relative', ("__get_by_name__", {"name_of_component" : "AdamOpimizerComponent"})),
                                               "input_for_fun_max_value" : 
                                                ('relative', 
                                                 [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]
                                                )


                                           }
                                       )                 
                                   }
                    )

                }),

                "discount_factor" : 0.98,
            
                "times_to_learn" : 20,
            
                "memory" : (TorchMemoryComponent, {
                    "capacity" : 256
                })
                
            
            }
    }

    return agent, agent_trainer