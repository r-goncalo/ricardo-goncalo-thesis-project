from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.core.input_management import InputSignature

from abc import abstractmethod

class Sampler(Component):
    
    '''
    Samples a Component
    '''
    
    parameters_signature = {
                    }    

    def sample(self) -> Component:
        pass