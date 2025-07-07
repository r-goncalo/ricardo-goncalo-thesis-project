from collections import namedtuple
from collections import deque
import random



from automl.component import Component, InputSignature, requires_input_proccess
from automl.ml.memory.memory_components import MemoryComponent

class DeqeueMemoryComponent(MemoryComponent):
    
    parameters_signature = {
                    }

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.memory = deque([], maxlen=self.capacity)

    #save a transition
    @requires_input_proccess
    def push(self, *args):
        self.memory.append(self.Transition(*args))

    @requires_input_proccess
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    @requires_input_proccess
    def sample_transposed(self, batch_size):
        return self.transpose(self.sample(batch_size))    
    
    
    def transpose(self, transitions):
        return self.Transition(*zip(*transitions))
    
    @requires_input_proccess
    def clear(self):
        self.memory.clear()
        
    @requires_input_proccess
    def get_all(self):
        return list(self.memory) 

    @requires_input_proccess
    def __len__(self):
        return len(self.memory)
    


