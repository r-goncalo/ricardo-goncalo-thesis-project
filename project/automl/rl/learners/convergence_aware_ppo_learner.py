from automl.basic_components.dynamic_value import get_value_or_dynamic_value
from automl.component import Component, ParameterSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentParameterSignature
from automl.core.advanced_input_utils import get_value_of_type_or_component
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.memory_utils import interpret_unit_values
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.ml.models.torch_model_utils import split_shared_params
from automl.rl.learners.learner_component import LearnerSchema

from automl.rl.learners.ppo_learner import PPOLearner
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automl.ml.models.torch_model_components import TorchModelComponent
from automl.rl.trainers.agent_trainer_component import AgentTrainer
import torch

from automl.utils.class_util import get_class_from

import torch.nn.functional as F

SHOULD_INITIALIZE_NEW_CRITIC = False

class ConvergenceAwarePPOLearner(PPOLearner):
    
    '''
    Proximal Policy Optimization Learner
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "kl_divergence_memory_size" : ParameterSignature(default_value=256),
                        
                        "kl_divergence_treshold" : ParameterSignature(default_value=0.001),
                        
                        "agent_trainer" : ComponentParameterSignature()

                        }    
    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        self.initialize_kl_divergence_memory()

        self.agent_trainer : AgentTrainer = self.get_input_value("agent_trainer")



    def initialize_kl_divergence_memory(self):

        self.kl_divergence_memory_size = self.get_input_value("kl_divergence_memory_size")

        # circular buffer
        self.kl_divergence_memory = []
        self._kl_memory_index = 0

        self.kl_divergence_treshold = self.get_input_value("kl_divergence_treshold")

        self.lg.writeLine(f"Convergence will be noted with kl divergence, using an average of {self.kl_divergence_memory_size} values, that must be bellow {self.kl_divergence_treshold}")




    def _update_convergence_and_check(self, new_log_probs, old_log_probs):
        """
        Computes approximate KL divergence between old and new policy.
        Stores it and checks convergence.
        """

        with torch.no_grad():
            # Standard PPO approximate KL
            approx_kl = (old_log_probs - new_log_probs).mean().item()

        # Fill buffer
        if len(self.kl_divergence_memory) < self.kl_divergence_memory_size:
            self.kl_divergence_memory.append(approx_kl)
        else:
            self.kl_divergence_memory[self._kl_memory_index] = approx_kl
            self._kl_memory_index = (self._kl_memory_index + 1) % self.kl_divergence_memory_size

        # Only check after buffer full
        if len(self.kl_divergence_memory) < self.kl_divergence_memory_size:
            return False

        avg_kl = sum(self.kl_divergence_memory) / self.kl_divergence_memory_size

        self.lg.writeLine(f"Average KL divergence: {avg_kl}")

        if avg_kl < self.kl_divergence_treshold:
            self.lg.writeLine(
                f"KL divergence {avg_kl} below threshold "
                f"{self.kl_divergence_treshold}. Convergence detected."
            )

            # Notify trainer
            self.agent_trainer.end_training()

            return True

        return False
    

    def _learn(self, trajectory: dict, discount_factor):

        log_prob_batch, new_log_probs = super()._learn(trajectory, discount_factor)

        self._update_convergence_and_check(new_log_probs, log_prob_batch)