

from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature, ComponentListInputSignature
from automl.core.input_management import InputSignature
from automl.ml.memory.memory_components import MemoryComponent


class MemorySampler(Component):

    '''Used with memory to do extra processing before sampling'''

    parameters_signature = {

    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()


    @requires_input_proccess
    def prepare(self, memory : MemoryComponent = None):
        '''Method to do any extra processing with whole memory before sampling'''

        self.memory : MemoryComponent = memory if memory is not None else self.memory

        if memory is None:
            raise Exception("No memory passed")
        

    @requires_input_proccess
    def sample(self, batch_size):
        return self.memory.sample(batch_size)
    
        
    @requires_input_proccess
    def get_all(self):
        '''Returns the total memory'''
        return self.memory.get_all()


    @requires_input_proccess
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        return self.memory.get_all_segmented(batch_size)


    @requires_input_proccess
    def __len__(self):
        return len(self.memory)