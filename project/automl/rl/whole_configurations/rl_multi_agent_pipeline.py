from automl.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
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
        "device" : "cuda",
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
                            FullyConnectedModelSchema, 
                            {
                            "hidden_layers" : 3,
                            "hidden_size" : 64
                            }
                            ),
                        }
                )
        },
        
        "rl_trainer" : (RLTrainerComponentParallel,
            
            {
            "num_episodes" : 100,
            
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval": 50,
                "times_to_learn" : 2,

                "learner" : (DeepQLearnerSchema, {
                               "target_update_rate" : 0.05,
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 0.0001
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "capacity" : 100
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