

import os

from automl.component import  requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.shapes_util import torch_shape_from_space
from automl.loggers.global_logger import globalWriteLine
import torch
from automl.ml.memory.memory_components import MemoryComponent

from automl.component import requires_input_proccess
from automl.core.input_management import InputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent


class TorchMemoryComponent(MemoryComponent, ComponentWithLogging):

    parameters_signature = {
        "capacity": InputSignature(default_value=1_000),
        "device": InputSignature(default_value="cuda"),
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position = 0
        self.total_size = 0


    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        self.device = self.get_input_value("device")

        self.lg.writeLine("Initializing TorchMemoryComponent...")

        self.allocate_computer_memory_to_transitions()
        
        self.position = 0
        self.total_size = 0
        
        self.lg.writeLine("TorchMemoryComponent initialized.")
        
        
        
    def allocate_computer_memory_to_transitions(self):
        
        '''Allocates the necessary space '''
        
        self.transitions = self._allocate_computer_memory_to_transitions_dictionary() # transitions saved in memory        


            
    def _allocate_computer_memory_to_transitions_dictionary(self) -> dict[str, torch.Tensor]:

        '''Allocates necessary space given the specification of name, shape and data type in field_shapes'''
            
        transitions : dict[str, torch.Tensor] = {}

        
        for field_name, field_shape, data_type in self.fields_shapes:

            field_shape = torch_shape_from_space(field_shape)

            #normally it would be state, action, next_state and reward 
            
            if data_type == None:
                   
                transitions[field_name] = torch.zeros((self.capacity, *field_shape),
                                                       device=self.device)                
            else:
                transitions[field_name] = torch.zeros((self.capacity, *field_shape),
                                                       device=self.device, dtype=data_type)
                
        for key in transitions.keys():
            self.lg.writeLine(f"Component {self.name} allocated for key '{key}' torch tensor with shape {transitions[key].shape}")
            
        return transitions
            
        

    @requires_input_proccess
    def push(self, transition):

        '''Pushes a transition into the saved transitions, possibly substituting another'''
        
        idx = self.position

        for field_name in self.field_names:
                        
            self.transitions[field_name][idx].copy_(transition[field_name])

        self.position = (self.position + 1) % self.capacity
        
        if self.total_size < self.capacity:
             self.total_size += 1
        
    

    @requires_input_proccess
    def sample(self, batch_size):

        '''Returns <batch_size> random elements from the saved transitions'''
        
        if len(self) < batch_size:
            raise ValueError("Not enough transitions to sample.")
        
        indices = torch.randint(0, self.total_size, (batch_size,), device=self.device)
                
        batch_data = {
            field_name: self.transitions[field_name][indices]
            for field_name in self.field_names
        }        
        
        return batch_data
    

    @requires_input_proccess
    def get_all(self):
        '''Returns the total memory'''

        batch_data = {
            field_name: self.transitions[field_name][:self.total_size]
            for field_name in self.field_names
        }
        
        return batch_data


    @requires_input_proccess
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        pass


        
    def _transitions_to_str(self, transitions_dict : dict[str, torch.Tensor], size, transitions_to_save):
        
        str_to_return = "" 
        
        for i in range(size): # for each row
        
            for field_name in transitions_to_save:    
            
                str_to_return += f"{field_name}: {transitions_dict[field_name][i]} "
                
            str_to_return += "\n"
            
        return str_to_return
            
            
         

    @requires_input_proccess
    def clear(self):
        
        '''Logicaly cleans the memory, without doing any deletion operation'''
        
        self.position = 0
        self.total_size = 0
        
        
        
    def __len__(self):
        return self.total_size
    
    def _save_state_internal(self):

        super()._save_state_internal()

        if self.total_size == 0:
            self.lg.writeLine("No transitions to save.")
            return

        self._save_transitions_to_disk()

    def _save_transitions_to_disk(self):

        transitions_dir = os.path.join(self.get_artifact_directory(), "transitions")

        os.makedirs(transitions_dir, exist_ok=True)

        self.lg.writeLine(f"Saving {self.total_size} transitions to {transitions_dir}")

        # Optional: remove old transition files
        for filename in os.listdir(transitions_dir):
            if filename.startswith("transition_") and filename.endswith(".pt"):
                os.remove(os.path.join(transitions_dir, filename))

        for idx in range(self.total_size):

            transition_dict = {}

            for field_name in self.field_names:
                # Always save on CPU (like torch models)
                transition_dict[field_name] = (
                    self.transitions[field_name][idx]
                    .detach()
                    .cpu()
                    .clone()
                )

            file_path = os.path.join(transitions_dir, f"transition_{idx}.pt")

            torch.save(transition_dict, file_path)

        self.lg.writeLine("Finished saving transitions.")


    def _load_state_internal(self):

        super()._load_state_internal()
        self._load_transitions_from_disk()

        

    def _load_transitions_from_disk(self):

        transitions_dir = os.path.join(self.get_artifact_directory(), "transitions")

        if not os.path.exists(transitions_dir):
            self.lg.writeLine("No transitions directory found.")
            return

        files = sorted(
            f for f in os.listdir(transitions_dir)
            if f.startswith("transition_") and f.endswith(".pt")
        )

        self.clear()

        for idx, filename in enumerate(files):

            transition_dict = torch.load(
                os.path.join(transitions_dir, filename),
                map_location=self.device
            )

            for field_name in self.field_names:
                self.transitions[field_name][idx].copy_(transition_dict[field_name])

            self.total_size += 1

        self.position = self.total_size % self.capacity

        self.lg.writeLine(f"Loaded {self.total_size} transitions from disk.")