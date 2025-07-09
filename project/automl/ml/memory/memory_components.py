from collections import namedtuple
from collections import deque
import random



from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, InputSignature, requires_input_proccess

class MemoryComponent(StatefulComponent):
    
    parameters_signature = {
                        "capacity" : InputSignature(default_value=1000),
                        "transition_data" : InputSignature(description="A list of data there is for the transitions, of tuples (name, shape)")
                    }
    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.capacity = self.input["capacity"]
        
        self.fields_shapes = self.input["transition_data"]
        
        self.field_names = [data_name for (data_name, _) in self.fields_shapes]
        
        self.Transition = namedtuple('Transition',
                                     self.field_names)
                          

    #save a transition
    @requires_input_proccess
    def push(self, transition):
        raise NotImplementedError()

    @requires_input_proccess
    def sample(self, batch_size):
        raise NotImplementedError()
    
    @requires_input_proccess
    def sample_transposed(self, batch_size):
        raise NotImplementedError()
    
    
    def transpose(self, transitions):
        raise NotImplementedError()
    
    @requires_input_proccess
    def clear(self):
        raise NotImplementedError()
        
    @requires_input_proccess
    def get_all(self):
        raise NotImplementedError()

    @requires_input_proccess
    def __len__(self):
        raise NotImplementedError()
    


