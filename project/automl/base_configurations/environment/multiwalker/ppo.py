from automl.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.ml.models.joint_model import ModelSequenceComponent
from automl.ml.models.model_initialization_strategy import TorchModelInitializationStrategyOrthogonal
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.ppo_learner import PPOLearner

from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.agent_trainer_ppo import AgentTrainerPPO
from automl.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel
from automl.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel

from automl.fundamentals.translator.tensor_translator import ToTorchTranslator


def config_dict():

    return {

    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",

    "input": {

        "do_full_setup_of_seed": True,
        "device": "cuda",

        "environment": (
            PettingZooEnvironmentWrapperParallel,
            {
                "environment": "multiwalker"
            }
        ),

        "agents_input": {

            "state_translator": (ToTorchTranslator, {}),

            "policy": (
                StochasticPolicy,
                {
                    "model": (
                        ModelSequenceComponent,
                        {
                            "name": "policy_model",
                            "models": [

                                [("__get_by_name__", {"name_of_component": "shared_model"})],

                                (FullyConnectedModelSchema,
                                {
                                    "name": "policy_head",
                                    "layers": [128, 64]
                                })

                            ]
                        }
                    )
                }
            )
        },

        "rl_trainer": (

            RLTrainerComponentParallel,

            {

            "name": "RLTrainerComponent",

            "limit_total_steps": 1_000_000,

            "predict_optimizations_to_do": True,

            "default_trainer_class": AgentTrainerPPO,

            "agents_trainers_input": {

                "optimization_interval": 2048,

                "times_to_learn": 10,

                "batch_size": 128,

                "discount_factor": 0.99,

                "learner": (

                    PPOLearner,

                    {

                        "lambda_gae": 0.95,

                        "clip_epsilon": 0.2,

                        "entropy_coef": 0.01,

                        "value_loss_coef": 0.5,

                        "critic_model": (

                            ModelSequenceComponent,

                            {

                                "name": "critic_model",

                                "output_shape": 1,

                                "models": [

                                    [("__get_by_name__", {"name_of_component": "shared_model"})],

                                    (FullyConnectedModelSchema,
                                    {
                                        "name": "critic_head",
                                        "layers": [128, 64]
                                    })

                                ]
                            }
                        ),

                        "optimizer": (

                            AdamOptimizer,

                            {

                                "name": "AdamOptimizerComponent",

                                "learning_rate": 3e-4,

                                "linear_decay_learning_rate_with_final_input_value_of":
                                ('relative',
                                    [
                                        ("__get_by_name__", {"name_of_component": "RLTrainerComponent"}),
                                        ("__get_exposed_value__", {
                                            "value_localization": ["optimizations_to_do_per_agent", "__any__"]
                                        })
                                    ]
                                ),

                                "clip_grad_value":

                                (
                                    DynamicLinearValueInRangeBasedOnComponent,
                                    {
                                        "input_for_fun_key": "optimizations_done",
                                        "initial_value": 0.5,
                                        "final_value": 0,
                                        "input_component": ('relative', ("__get_by_name__", {"name_of_component": "AdamOptimizerComponent"})),

                                        "input_for_fun_max_value":
                                        ('relative',
                                            [
                                                ("__get_by_name__", {"name_of_component": "RLTrainerComponent"}),
                                                ("__get_exposed_value__", {
                                                    "value_localization": ["optimizations_to_do_per_agent", "__any__"]
                                                })
                                            ]
                                        )
                                    }
                                )
                            }

                        )
                    }
                ),

                "memory": (

                    TorchMemoryComponent,

                    {
                        "device": "cpu",
                        "capacity": 2048
                    }

                )

            }

            }

        )

    },

    "child_components": [

        {

            "__type__": FullyConnectedModelSchema,

            "name": "shared_model",

            "input": {

                "layers": [256, 256]

            }

        }

    ]

}