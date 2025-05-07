


from automl.component import Component
from automl.utils.json_component_utils import json_string_of_component, component_from_json_string
from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.core.artifact_management import ArtifactComponent
import os



class StatefulComponent(ArtifactComponent):
    
    '''A component with the capability of storing its state in its respective directory and later load it'''
    
    def __save_state_recursive_child_components(self, child_components : list[Component]):
        
        '''Saves state of state of child components'''
        
        for child_component in child_components:
                        
            self.__save_state_recursive_child_components(child_component.child_components)
            
            if isinstance(child_component, StatefulComponent):

                child_component.save_state(recursive=False, save_definition=False)
        
    
    
    def save_state(self, recursive=True, save_definition=True):
        '''Saves the state of this component and child components'''
    
        if recursive:
            self.__save_state_recursive_child_components(self.child_components)
            
        if save_definition:
            
            json_str = json_string_of_component(self, save_exposed_values=True)
        
            write_text_to_file(self.artifact_directory, 'definition.json', json_str)  
    
    
    
    
    def load_state(folder_path): #note this is not a method 
        
        json_str = read_text_from_file(folder_path, 'definition.json')
        
        component_to_return = component_from_json_string(json_string=json_str)
        
        if not isinstance(component_to_return, StatefulComponent):
            raise Exception("Tried to load state of component which is not a stateful component")
        
        return component_to_return
