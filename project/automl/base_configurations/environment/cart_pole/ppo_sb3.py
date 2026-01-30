



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

from automl.fundamentals.translator.tensor_translator import ToTorchTranslator
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.environment.gymnasium.aec_gymnasium_env import AECGymnasiumEnvironmentWrapper
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
        "environment": (AECGymnasiumEnvironmentWrapper, {"environment" : "CartPole-v1"}),
        "agents_input": {

            "policy" : ( StochasticPolicy,
                        {
                        "model" : (
                            FullyConnectedModelSchema, 
                            {
                            "layers" : [64, 64, 64]
                            }
                            ),
                        }
                ),

            "state_translator" : (ToTorchTranslator, {})
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

def hyperparameter_suggestions():

    from automl.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automl.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automl.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion

    return [

        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion= ("int",  {"low" : 64, "high" : 1024 }),
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "optimization_interval"],
                ["input", "rl_trainer", 1, "agents_trainers_input", "memory", 1, "capacity"],
            ]

        ),

        DisjointHyperparameterSuggestion(
            name = "clip_grad_value_strat",
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "optimizer", 1, "clip_grad_value"]
            ],
            disjoint_hyperparameter_suggestions= [
                SingleHyperparameterSuggestion(
                    name="clip_grad_value",
                    value_suggestion=("float", {"low" : 0.05, "high" : 1.0})
                ),
                ComplexHpSuggestion(
                    "clip_grad_dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key" : "optimizations_done",
                            "initial_value" : 0.2,
                            "final_value" : 0,
                            "input_component" : ('relative', [("__get_by_name__", {"name_of_component" : "AdamOpimizerComponent"})]),
                            "input_for_fun_max_value" : 
                             ('relative', 
                              [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]
                             )
                        }
                    ],
                    actual_hyperparameter_suggestion= SingleHyperparameterSuggestion(
                        name="clip_grad_dynamic_value",
                        value_suggestion=("float", {"low" : 0.05, "high" : 0.5}),
                        hyperparameter_localizations=[[1, "initial_value"]]
                    )

                )

            ]
        )

    ]



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