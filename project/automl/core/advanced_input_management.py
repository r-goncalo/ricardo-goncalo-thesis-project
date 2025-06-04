


'''This module provides advanced input management, importing from input_management and component'''
        

from automl.component import Component
from automl.core.input_management import InputSignature
from automl.utils.json_component_utils import gen_component_from


class ComponentInputSignature(InputSignature):
    
    '''Abstracts the passage of components in other components inputs'''
    
    def get_component_from_input(component_with_input : Component, key):
        
        '''Returns a component from a CompoenentsInputSignature passed value'''
        
        value = component_with_input.input[key]
        
        if isinstance(value, Component):
            component = value
        
        else:
        
            component = gen_component_from(value)
            component_with_input.define_component_as_child(component)

        return component
    
    
    
    def __init__(default_component_definition = None):
        
        def generator(self : Component):
            
            if default_component_definition is not None:
                return ComponentInputSignature.get_component_from_input(self, default_component_definition)
            
            else:
                return None
        
        super().__init__(possible_types=[type, dict, str, tuple])
        
    

            
        