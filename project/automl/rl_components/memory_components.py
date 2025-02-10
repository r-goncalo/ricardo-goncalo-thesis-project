from collections import namedtuple
from collections import deque
import random



from ..component import Component, InputSignature, requires_input_proccess

class MemoryComponent(Component):
    
    parameters_signature = {
                        "capacity" : InputSignature(default_value=1000)
                    }
    
    #defines the format used to store states, actions, next_states and rewards
    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def proccess_input(self):
        
        super().proccess_input()
        
        self.capacity = self.input["capacity"]                  
        self.memory = deque([], maxlen=self.capacity)

    #save a transition
    @requires_input_proccess
    def push(self, *args):
        self.memory.append(self.Transition(*args))

    @requires_input_proccess
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    @requires_input_proccess
    def __len__(self):
        return len(self.memory)