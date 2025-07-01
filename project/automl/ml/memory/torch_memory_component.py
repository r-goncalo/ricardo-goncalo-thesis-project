

from collections import namedtuple
from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.shapes_util import discrete_output_layer_size_of_space
import torch
from automl.ml.memory.memory_components import MemoryComponent


class TorchMemoryComponent(MemoryComponent, ComponentWithLogging, StatefulComponent):
    
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
        
        print(f"State dim: {self.state_dim}")
        print(f"Action dim: {self.action_dim}")
        
        self.lg.writeLine("Initializing torch memory component...")
        
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device=self.device)
            mem_reserved = torch.cuda.memory_reserved(device=self.device)
            
            self.lg.writeLine(f"Current memory allocated in device: {mem_alloc} MB")
            self.lg.writeLine(f"Current memory reserved in the device: {mem_reserved} MB")
        

        # Pre-allocate memory on GPU
        self.states = torch.zeros((self.capacity, *self.state_dim), 
                                  device=self.device, dtype=self.dtype)
        self.actions = torch.zeros((self.capacity, discrete_output_layer_size_of_space(self.action_dim)), # TODO: this assumes a single value comming from this function
                                   device=self.device, dtype=self.dtype)
        self.next_states = torch.zeros((self.capacity, *self.state_dim), 
                                       device=self.device, dtype=self.dtype)
        self.rewards = torch.zeros((self.capacity, 1), 
                                   device=self.device, dtype=self.dtype)
        
        self.position = 0
        self.size = 0
        

        self.lg.writeLine("Finished initializing torch memory component...")
        
        if torch.cuda.is_available():
            new_mem_alloc = torch.cuda.memory_allocated(device=self.device)
            new_mem_reserved = torch.cuda.memory_reserved(device=self.device)
            
            dif_mem_alloc = new_mem_alloc - mem_alloc
            dif_mem_reserved = new_mem_reserved - mem_reserved
            
            self.lg.writeLine(f"Current memory allocated in device: {new_mem_alloc} B, difference of {dif_mem_alloc / 1024} MB")
            self.lg.writeLine(f"Current memory reserved in the device: {new_mem_reserved} B, difference of {dif_mem_reserved / 1024} MB")

    @requires_input_proccess
    def push(self, state, action, next_state, reward):
        idx = self.position

        self.states[idx].copy_(state)
        self.actions[idx] = action
        self.next_states[idx].copy_(next_state)
        self.rewards[idx] = reward

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    @requires_input_proccess
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size - 1,), device=self.device)

        batch = self.Transition(
            state=self.states[indices],
            action=self.actions[indices],
            next_state=self.next_states[indices],
            reward=self.rewards[indices],
        )
        return batch # TODO: this has a different form of normal memory sample, it is wrong
    
    @requires_input_proccess
    def sample_transposed(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size - 1,), device=self.device)

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