
from automl.basic_components.exec_component import InputSignature
from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature, ComponentListInputSignature
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.memory_samplers.memory_sampler import MemorySampler
from automl.ml.memory.memory_utils import interpret_values
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.learners.ppo_learner import PPOLearner
import torch


class PPOAdvantagesCalcSampler(MemorySampler):

    '''
    Calculates and normalizes values for PPO before mini batch sampling
    It assumes the memory will be cleared after, and so it does in_place transformations
    '''

    parameters_signature = {
        "learner" : ComponentInputSignature(),
        "discount_factor" : InputSignature()
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.learner : PPOLearner = self.get_input_value("learner")

        if not isinstance(self.learner, PPOLearner):
            raise Exception("Assumes PPO Learner")
        
        self.discount_factor = self.get_input_value("discount_factor")

    def prepare(self, memory : MemoryComponent = None):
        '''Does the in place transformations'''
        
        super().prepare(memory)

        self.memory : TorchMemoryComponent = self.memory

        self.device = memory.device

        state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch, critic_pred_batch = self.learner.interpret_trajectory(self.memory.get_all())

        with torch.no_grad():
            values, next_values = self.learner.compute_values_estimates(state_batch, action_batch, next_state_batch, done_batch)
            values_error, non_normalized_advantages, advantages, returns = self.learner.compute_error_and_advantage(self.discount_factor, reward_batch, next_values, values, done_batch)

        self.extra_memory = {
            "values" : interpret_values(values, self.device),
            "next_values" : interpret_values(next_values, self.device),
            "advantages" : interpret_values(advantages, self.device),
            "returns" : interpret_values(returns, self.device)
        }

        self.field_names = [*memory.field_names, *self.extra_memory.keys()]
        self.transitions = {**self.memory.transitions, **self.extra_memory}

    @requires_input_proccess
    def sample(self, batch_size):

        '''Returns <batch_size> random elements from the saved transitions'''
        
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")
        
        indices = torch.randint(0, len(self.memory), (batch_size,), device=self.memory.device)
                
        batch_data = {
            field_name: self.transitions[field_name][indices]
            for field_name in self.field_names
        }        
        
        return batch_data
    
    @requires_input_proccess
    def get_all(self):
        '''Returns the total memory'''

        batch_data = {
            field_name: self.transitions[field_name][:len(self.memory)]
            for field_name in self.field_names
        }
        
        return batch_data