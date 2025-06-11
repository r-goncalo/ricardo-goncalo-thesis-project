


'''This module provides advanced input management, importing from input_management and component'''
        

from automl.component import Component
from automl.core.input_management import InputSignature
from automl.utils.json_component_utils import gen_component_from


class ComponentInputSignature(InputSignature):
    
    '''Abstracts the passage of components in other components inputs'''
    
    def get_component_from_input(component_with_input : Component, key):
        
        '''Returns a component from a ComponentInputSignature passed value'''
        
        value = component_with_input.input[key]
        
        if isinstance(value, Component):
            component = value
        
        else:
        
            component = gen_component_from(value)
            component_with_input.define_component_as_child(component)

        return component
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
        
            def generator(self : Component): # will return the component to be saved in 
                
                print("generator for evaluater component called")
                print(default_component_definition)

                component = gen_component_from(default_component_definition)
                self.define_component_as_child(component)
                
                return component
            
            super().__init__(possible_types=[Component, type, dict, str, tuple], generator=generator, **kwargs)
        
        else:
            super().__init__(possible_types=[Component, type, dict, str, tuple], **kwargs)
            

        
        
        
    

            
        