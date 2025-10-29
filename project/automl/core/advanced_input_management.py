


'''This module provides advanced input management, importing from input_management and component'''
        

from automl.component import Component
from automl.core.input_management import InputSignature
from automl.utils.json_utils.json_component_utils import gen_component_from
from automl.core.localizations import get_component_by_localization


CHANGE_INPUT = True

class LookableInputSignature(InputSignature):

    '''An input with functionality that allows it to look for its value'''

    def get_value_from_input(component_with_input : Component, key, possible_types):
        
        value = InputSignature.get_value_from_input(component_with_input, key)

        if value == None:
            return None

        elif isinstance(value, possible_types):
            return possible_types

        else: # is localization

            try:
                return get_component_by_localization(component_with_input, value)
            
            except Exception as e:

                raise Exception(f"Exception when trying to get value for key {key} for component {component_with_input.name}, with assumed localization {value}: \n{e}") from e


class ComponentInputSignature(InputSignature):
    
    '''Abstracts the passage of components in other components inputs'''

    
    possible_types = [Component, type, dict, str, tuple, list]
    
    def get_value_from_input(component_with_input : Component, key, input_if_generated=None):
        
        '''Returns a component from a ComponentInputSignature passed value'''
        
        value = InputSignature.get_value_from_input(component_with_input, key)

        if value is None:
            return value
        
        component = gen_component_from(value, component_with_input, input_if_generated)

        if CHANGE_INPUT:
            component_with_input.input[key] = component

        return component
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
        
        if "possible_types" in kwargs.keys():
            kwargs["possible_types"] = [*ComponentInputSignature.possible_types, *kwargs["possible_types"]]

        else:
            kwargs["possible_types"] = ComponentInputSignature.possible_types
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
        
            def generator(self : Component): # will return the component to be saved in 
                
                component = gen_component_from(default_component_definition)
                self.define_component_as_child(component)
                
                return component
            
            super().__init__(generator=generator, **kwargs)
        
        else:
            super().__init__(**kwargs)            
            


class ComponentListInputSignature(InputSignature):
    
    '''Abstracts the passage of component list in other components inputs'''
    
    def get_value_from_input(component_with_input : Component, key) -> list[Component]:
        
        '''Returns a component list from a ComponentListInputSignature passed value'''
        
        list_of_components = component_with_input.input[key]
        
        to_return : list[Component] = []

        for value in list_of_components:
        
            if isinstance(value, Component):
                to_return.append(value)

            else:

                component = gen_component_from(value, component_with_input)
                to_return.append(component)

        if CHANGE_INPUT:
            component_with_input.input[key] = to_return

        return to_return
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
            
            raise NotImplementedError("Default component definition is not implemented for component lists")
        
        else:
            super().__init__(**kwargs)
            
            
class ComponentDictInputSignature(InputSignature):
    
    '''Abstracts the passage of component list in other components inputs'''
    
    def get_value_from_input(component_with_input : Component, key) -> dict[Component]:
        
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

        if CHANGE_INPUT:
            component_with_input.input[key] = dict_of_components

        return dict_of_components
    
    
    
    def __init__(self, default_component_definition = None, **kwargs):
        
        '''Default component definition can be a component, a json string, a dictionary, and so on'''
    
        if default_component_definition is not None and "generator" in kwargs.keys():
            raise Exception("Geneator in arguments of Component Input Signature when there is a default component definition")
        

        if default_component_definition is not None:
            
            raise NotImplementedError("Default component definitions in component dicts is not implemented")
        
        else:
            super().__init__(**kwargs)
            

        
        
        
    

            
        