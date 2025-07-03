

from collections import namedtuple
from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.maths import nearest_highest_multiple, nearest_multiple
from automl.utils.shapes_util import discrete_output_layer_size_of_space
import torch
from automl.ml.memory.memory_components import MemoryComponent


import os
import torch
import numpy as np
from pathlib import Path

from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.shapes_util import discrete_output_layer_size_of_space
from automl.ml.memory.memory_components import MemoryComponent


class TorchDiskMemoryComponent(MemoryComponent, ComponentWithLogging, StatefulComponent):

    parameters_signature = {
        "capacity": InputSignature(default_value=1_000),
        "state_dim": InputSignature(),
        "action_dim": InputSignature(),
        "device": InputSignature(default_value="cuda"),
        "dtype": InputSignature(default_value=torch.float32),
        "max_in_memory": InputSignature(default_value=80),
        "storage_dir": InputSignature(default_value="./memory_storage"),
    }

    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.state_dim = self.input["state_dim"]
        self.action_dim = self.input["action_dim"]
        self.device = self.input["device"]
        self.dtype = self.input["dtype"]
        self.max_in_memory = self.input["max_in_memory"]

        self.set_capaticy()

        self.lg.writeLine("Initializing hybrid TorchMemoryComponent...")

        self.allocate_computer_memory_to_transitions()

        self.position = 0
        self.size_in_memory = 0
        self.total_size = 0
        
        self.initialize_disk_files()

        self.lg.writeLine("TorchMemoryComponent initialized.")

    def set_capaticy(self):
        
        '''Sets capacity to an appropriate value, multiple of the max transitions in memory'''

        self.capacity = nearest_highest_multiple(self.input["capacity"], self.max_in_memory)
        
        if self.capacity != self.input["capacity"]:
            self.lg.writeLine(f"Capacity of {self.input['capacity']} was changed to {self.capacity} due to it not being a multiple of max in memory ({self.max_in_memory})")

        
    def allocate_computer_memory_to_transitions(self):
        
        '''Allocates the necessary space '''
        
        self.states = torch.zeros((self.max_in_memory, *self.state_dim),
                                  device=self.device, dtype=self.dtype)
        self.actions = torch.zeros((self.max_in_memory, discrete_output_layer_size_of_space(self.action_dim)),
                                   device=self.device, dtype=self.dtype)
        self.next_states = torch.zeros((self.max_in_memory, *self.state_dim),
                                       device=self.device, dtype=self.dtype)
        self.rewards = torch.zeros((self.max_in_memory, 1),
                                   device=self.device, dtype=self.dtype)
        
        
    def initialize_disk_files(self):
        
        self.storage_dir = Path(os.path.join(self.artifact_directory, self.input["storage_dir"]))

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.disk_file_position = 0
        
        number_of_times_size_in_memory_in_capacity = int(self.capacity / self.max_in_memory)
        
        self.lg.writeLine(f"There are {number_of_times_size_in_memory_in_capacity} times the max_in_memory in the whole capacity of the memory")
        
        self.number_of_files = number_of_times_size_in_memory_in_capacity - 1
                
        self.disk_files = [os.path.join(self.storage_dir, f"transitions_{i}.pt") for i in range(0, self.number_of_files)]
        

    @requires_input_proccess
    def push(self, state, action, next_state, reward):
        idx = self.position

        self.states[idx].copy_(state)
        self.actions[idx].copy_(action)
        self.next_states[idx].copy_(next_state)
        self.rewards[idx].copy_(reward)

        self.position = (self.position + 1) % self.max_in_memory
        self.total_size += 1

        if self.size_in_memory < self.max_in_memory:
            self.size_in_memory += 1
    
            
        else:
            # Buffer is full → flush to disk
            self._flush_to_disk()

    

    def _flush_to_disk(self):
        
        """
        Save current memory buffer to disk and clear memory.
        """
        file_path = self.disk_files[self.disk_file_position]
        
        torch.save({
            "state": self.states.cpu(),
            "action": self.actions.cpu(),
            "next_state": self.next_states.cpu(),
            "reward": self.rewards.cpu(),
            "size": self.size_in_memory
        }, file_path)

        self.lg.writeLine(f"Flushed {self.size_in_memory} transitions to disk at {file_path}")
        
        self.disk_file_position = (self.disk_file_position + 1) % self.number_of_files

        # We don't clear the memory because we don't need to, only new positions will be used
        self.position = 0
        self.size_in_memory = 0
        
        
        if self.total_size == self.capacity:
        
            self.total_size -= self.size_in_memory #we deleted self.size_in_memory entries previously stored
        
    

    @requires_input_proccess
    def sample(self, batch_size):
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")

        # Probability of sampling from disk vs memory
        prob_disk = len(self.disk_files) / (len(self.disk_files) + 1e-5)
        from_disk = False

        if len(self.disk_files) > 0:
            from_disk = torch.rand(1).item() < prob_disk

        if from_disk:
            return self._sample_from_disk(batch_size)
        else:
            return self._sample_from_memory(batch_size)

    def _sample_from_memory(self, batch_size):
        
        indices = torch.randint(0, self.size_in_memory, (batch_size,), device=self.device)

        batch = self.Transition(
            state=self.states[indices],
            action=self.actions[indices],
            next_state=self.next_states[indices],
            reward=self.rewards[indices]
        )
        return batch

    def _sample_from_disk(self, batch_size):
        
        file_path = np.random.choice(self.disk_files)

        data = torch.load(file_path, map_location="cpu")
        size = data["size"]

        indices = torch.randint(0, size, (batch_size,))
        batch = self.Transition(
            state=data["state"][indices].to(self.device, dtype=self.dtype),
            action=data["action"][indices].to(self.device, dtype=self.dtype),
            next_state=data["next_state"][indices].to(self.device, dtype=self.dtype),
            reward=data["reward"][indices].to(self.device, dtype=self.dtype)
        )
        return batch

    @requires_input_proccess
    def sample_transposed(self, batch_size):
        batch = self.sample(batch_size)
        return self.Transition(*batch)