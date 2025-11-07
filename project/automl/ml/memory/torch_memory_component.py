

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
        

        batch = self.Transition(**batch_data)
        
        return batch



    @requires_input_proccess
    def sample_transposed(self, batch_size):
        batch = self.sample(batch_size)
        return self.Transition(*batch)
        
        
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