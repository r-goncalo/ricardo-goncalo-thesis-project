from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.ml.models.conv_model import ConvModel
from automl.ml.models.dynamic_conv_model import DynamicConvModelSchema
from automl.ml.models.joint_model import ModelSequenceComponent
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.q_learner import DeepQLearnerSchema
from automl.rl.policy.qpolicy import QPolicy

from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.trainers.agent_trainer_component_dqn import AgentTrainerDQN
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel
from automl.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel
from automl.fundamentals.translator.torch_image_state_translator import ImageReverterToSingleChannel, ImageNormalizer
from automl.fundamentals.translator.translator import TranslatorSequence
from automl.fundamentals.translator.tensor_translator import ToTorchTranslator


def config_dict():


    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cpu",
        "environment": (PettingZooEnvironmentWrapperParallel, {
                            "environment": "cooperative_pong"}),
        "agents_input": {
            "state_memory_size" : 2,

            "state_translator" : (TranslatorSequence, {
                "translators_sequence" : [
                        (ToTorchTranslator, {}),
                        (ImageReverterToSingleChannel, {}),
                        (ImageNormalizer, {})
                    ],
                    "in_place_translation" : True
                },
            
            ),

            "policy" : ( QPolicy,
                        {

                        "model" : (
                            ModelSequenceComponent,
                             {"models": [ 

                                [("__get_by_name__", {"name_of_component" : "shared_model"})], 
                                
                                (FullyConnectedModelSchema, 
                                {
                                "device" : "cuda",
                                "layers" : [64, 32]
                                }
                                ),
                             ]}

                        )
                        }
            )
        },
        
        "rl_trainer" : (RLTrainerComponentParallel,
            
            {
            "limit_total_steps" : 10e4,
            
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval": 1000,
                "times_to_learn" : 64,
                "batch_size" : 64,

                "learner" : (DeepQLearnerSchema, {
                                "device" : "cuda",
                               "target_update_rate" : 1.0,
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 0.00001
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "device" : "cpu",
                    "capacity" : 1000
                }),
                
                "exploration_strategy" : (EpsilonGreedyStrategy,
                                                                  {
                                            "epsilon_end" : 0.04,
                                            "epsilon_start" : 1.0,
                                            "epsilon_decay" : 0.16
                                                                  }
                                          )
    
                    }

            }
        )
        
    },
    "child_components" : [
        {
            "__type__" : DynamicConvModelSchema,
            "name" : "shared_model",
            "input" : {
                "cnn_layers" : [
                    {"kernel_size" : 8, "out_channels" : 32, "stride" : 4},
                    {"kernel_size" : 4, "out_channels" : 64, "stride" : 2},
                    {"kernel_size" : 3, "out_channels" : 64, "stride" : 1},
                    

                ]
            }
        }
    ]
}