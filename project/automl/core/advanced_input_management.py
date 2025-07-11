


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
                
                component = gen_component_from(default_component_definition)
                self.define_component_as_child(component)
                
                return component
            
            super().__init__(possible_types=[Component, type, dict, str, tuple], generator=generator, **kwargs)
        
        else:
            super().__init__(possible_types=[Component, type, dict, str, tuple], **kwargs)
            
            


class ComponentListInputSignature(InputSignature):
    
    '''Abstracts the passage of component list in other components inputs'''
    
    def get_component_list_from_input(component_with_input : Component, key) -> list[Component]:
        
        '''Returns a component list from a ComponentListInputSignature passed value'''
        
        list_of_components = component_with_input.input[key]
        
        to_return : list[Component] = []

        for value in list_of_components:
        
            if isinstance(value, Component):
                to_return.append(value)

            else:
            
                component = gen_component_from(value)
                component_with_input.define_component_as_child(component)

                to_return.append(component)

        return list_of_components
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
            
            raise NotImplementedError("Default component definition is not implemented for component lists")
            
            #(component_definition, n_components) = default_component_definition
        
            #def generator(self : Component): # will return the component to be saved in 

            #    list_of_components : list[Component] = [None] * n_components

            #    for i in range(n_components):

            #        component = gen_component_from(component_definition)
            #        self.define_component_as_child(component)
            #        
            #        list_of_components[i] = component
            #    
            #    return list_of_components
            #
            #super().__init__(generator=generator, **kwargs)
        
        else:
            super().__init__(**kwargs)
            
            
class ComponentDictInputSignature(InputSignature):
    
    '''Abstracts the passage of component list in other components inputs'''
    
    def get_component_list_from_input(component_with_input : Component, key) -> dict[Component]:
        
        '''Returns a component list from a ComponentListInputSignature passed value'''
        
        dict_of_components : dict = component_with_input.input[key]
        
        to_return : dict[Component] = {}

        for key, value in dict_of_components.values():
        
            if isinstance(value, Component):
                to_return[key] = value

            else:
            
                component = gen_component_from(value)
                component_with_input.define_component_as_child(component)

                to_return[key] = component

        return dict_of_components
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
            
            raise NotImplementedError("Default component definitions in component dicts is not implemented")
        
        else:
            super().__init__(**kwargs)
            

        
        
        
    

            
        