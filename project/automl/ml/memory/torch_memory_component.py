

from collections import namedtuple
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
import torch
from automl.ml.memory.memory_components import MemoryComponent


class TorchMemoryComponent(MemoryComponent):
    
    parameters_signature = {
        "capacity": InputSignature(default_value=1000),
        "state_dim": InputSignature(),      # e.g. 4
        "action_dim": InputSignature(),     # e.g. 1
        "device": InputSignature(default_value="cuda"),
        "dtype": InputSignature(default_value=torch.float32)
    }


    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.capacity = self.input["capacity"]
        self.state_dim = self.input["state_dim"]
        self.action_dim = self.input["action_dim"]
        self.device = self.input["device"]
        self.dtype = self.input["dtype"]

        # Pre-allocate memory on GPU
        self.states = torch.zeros((self.capacity, self.state_dim), 
                                  device=self.device, dtype=self.dtype)
        self.actions = torch.zeros((self.capacity, self.action_dim), 
                                   device=self.device, dtype=self.dtype)
        self.next_states = torch.zeros((self.capacity, self.state_dim), 
                                       device=self.device, dtype=self.dtype)
        self.rewards = torch.zeros((self.capacity, 1), 
                                   device=self.device, dtype=self.dtype)
        
        self.position = 0
        self.size = 0

    @requires_input_proccess
    def push(self, state, action, next_state, reward):
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.next_states[idx] = next_state
        self.rewards[idx] = reward

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    @requires_input_proccess
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        batch = self.Transition(
            state=self.states[indices],
            action=self.actions[indices],
            next_state=self.next_states[indices],
            reward=self.rewards[indices],
        )
        return batch

    @requires_input_proccess
    def clear(self):
        self.position = 0
        self.size = 0

    @requires_input_proccess
    def get_all(self):
        """
        Return all stored transitions as a Transition tuple of tensors.
        """
        idxs = torch.arange(0, self.size, device=self.device)
        return self.Transition(
            state=self.states[idxs],
            action=self.actions[idxs],
            next_state=self.next_states[idxs],
            reward=self.rewards[idxs]
        )

    @requires_input_proccess
    def __len__(self):
        return self.size