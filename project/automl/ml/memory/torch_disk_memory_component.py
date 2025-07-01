

from collections import namedtuple
from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
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


class TorchMemoryComponent(MemoryComponent, ComponentWithLogging, StatefulComponent):

    parameters_signature = {
        "capacity": InputSignature(default_value=1_000),
        "state_dim": InputSignature(),
        "action_dim": InputSignature(),
        "device": InputSignature(default_value="cuda"),
        "dtype": InputSignature(default_value=torch.float32),
        "max_in_memory": InputSignature(default_value=100),
        "storage_dir": InputSignature(default_value="./memory_storage"),
    }

    def proccess_input_internal(self):
        super().proccess_input_internal()

        self.capacity = self.input["capacity"]
        self.state_dim = self.input["state_dim"]
        self.action_dim = self.input["action_dim"]
        self.device = self.input["device"]
        self.dtype = self.input["dtype"]
        self.max_in_memory = self.input["max_in_memory"]
        self.storage_dir = os.path.join(self.artifact_directory, self.input["storage_dir"])

        self.lg.writeLine("Initializing hybrid TorchMemoryComponent...")

        # Allocate in-memory buffer
        self.states = torch.zeros((self.max_in_memory, *self.state_dim),
                                  device=self.device, dtype=self.dtype)
        self.actions = torch.zeros((self.max_in_memory, discrete_output_layer_size_of_space(self.action_dim)),
                                   device=self.device, dtype=self.dtype)
        self.next_states = torch.zeros((self.max_in_memory, *self.state_dim),
                                       device=self.device, dtype=self.dtype)
        self.rewards = torch.zeros((self.max_in_memory, 1),
                                   device=self.device, dtype=self.dtype)

        self.position = 0
        self.size_in_memory = 0
        self.total_size = 0

        # Disk storage setup
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.disk_file_counter = 0
        self.disk_files = []  # List of file paths on disk

        self.lg.writeLine("TorchMemoryComponent initialized.")

    @requires_input_proccess
    def push(self, state, action, next_state, reward):
        idx = self.position

        self.states[idx].copy_(state)
        self.actions[idx].copy_(action)
        self.next_states[idx].copy_(next_state)
        self.rewards[idx].copy_(reward)

        self.position = (self.position + 1) % self.max_in_memory

        if self.size_in_memory < self.max_in_memory:
            self.size_in_memory += 1
        else:
            # Buffer is full → flush to disk
            self._flush_to_disk()

        self.total_size += 1

    def _flush_to_disk(self):
        """
        Save current memory buffer to disk and clear memory.
        """
        file_path = self.storage_dir / f"transitions_{self.disk_file_counter}.pt"
        torch.save({
            "state": self.states.cpu(),
            "action": self.actions.cpu(),
            "next_state": self.next_states.cpu(),
            "reward": self.rewards.cpu(),
            "size": self.size_in_memory
        }, file_path)

        self.lg.writeLine(f"Flushed {self.size_in_memory} transitions to disk at {file_path}")
        self.disk_files.append(file_path)
        self.disk_file_counter += 1

        # Clear memory buffer
        self.states.zero_()
        self.actions.zero_()
        self.next_states.zero_()
        self.rewards.zero_()
        self.position = 0
        self.size_in_memory = 0

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

    @requires_input_proccess
    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.next_states.zero_()
        self.rewards.zero_()
        self.position = 0
        self.size_in_memory = 0
        self.total_size = 0

        for f in self.disk_files:
            os.remove(f)
        self.disk_files = []
        self.disk_file_counter = 0

    @requires_input_proccess
    def get_all(self):
        """
        Returns all transitions (from memory + disk) merged as tensors.
        Warning: may consume a lot of RAM if dataset is huge.
        """
        # Gather in-memory
        result_states = [self.states[:self.size_in_memory]]
        result_actions = [self.actions[:self.size_in_memory]]
        result_next_states = [self.next_states[:self.size_in_memory]]
        result_rewards = [self.rewards[:self.size_in_memory]]

        # Gather from disk
        for f in self.disk_files:
            data = torch.load(f, map_location=self.device)
            result_states.append(data["state"].to(self.device))
            result_actions.append(data["action"].to(self.device))
            result_next_states.append(data["next_state"].to(self.device))
            result_rewards.append(data["reward"].to(self.device))

        return self.Transition(
            state=torch.cat(result_states, dim=0),
            action=torch.cat(result_actions, dim=0),
            next_state=torch.cat(result_next_states, dim=0),
            reward=torch.cat(result_rewards, dim=0)
        )

    @requires_input_proccess
    def __len__(self):
        total_disk_size = sum(torch.load(f)["size"] for f in self.disk_files)
        return total_disk_size + self.size_in_memory