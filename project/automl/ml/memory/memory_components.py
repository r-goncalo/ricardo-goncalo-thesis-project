from collections import namedtuple
from collections import deque
import random



from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, InputSignature, requires_input_proccess

class MemoryComponent(StatefulComponent):
    
    parameters_signature = {
                        "capacity" : InputSignature(default_value=1000),
                        "transition_data" : InputSignature(description="A list of data there is for the transitions, of tuples (name, shape, (type)?)")
                    }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.capacity = self.get_input_value("capacity")
        
        self.fields_shapes = []

        self.transition_data = self.get_input_value("transition_data")
        
        for transition_data in self.transition_data:
            
            if len(transition_data) < 3:
                data_name, shape = transition_data
                data_type = None
            
            else:
                data_name, shape, data_type = transition_data
                
            self.fields_shapes.append((data_name, shape, data_type))
                
        self.field_names = [data_name for (data_name, _, _) in self.fields_shapes]
        
                          

    #save a transition
    @requires_input_proccess
    def push(self, transition):
        raise NotImplementedError()

    @requires_input_proccess
    def sample(self, batch_size):
        raise NotImplementedError()
    
    
    @requires_input_proccess
    def clear(self):
        raise NotImplementedError()
        
    @requires_input_proccess
    def get_all(self):
        '''Returns the total memory'''
        pass


    @requires_input_proccess
    def get_all_segmented(self, batch_size):
        '''Returns ordered list of segmented memory with batch_size'''
        pass


    @requires_input_proccess
    def __len__(self):
        raise NotImplementedError()
    


