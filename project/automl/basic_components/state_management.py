


from typing import final
from automl.component import Component, requires_input_proccess
from automl.utils.json_component_utils import decode_components_from_dict, dict_from_json_string, gen_component_from, json_string_of_component, component_from_json_string, set_values_of_dict_in_component
from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.basic_components.artifact_management import ArtifactComponent
import os

from automl.consts import CONFIGURATION_FILE_NAME

import weakref
import gc
                
                
                
                

class StatefulComponent(ArtifactComponent):
    
    '''
    A component with the capability of storing its state in its respective directory and later load it
    All classes which extend this class should implement the save_state method and load_state function
    
    Note that all components can inherently store and load their state if it is purely their input, output and exposed values
    '''
    
    @final
    def save_state(self, save_definition=False):
        '''
        Saves the state of this component        
        This method should not typically be called directly, but rather through the save_component_with_state_to_folder function
        '''
            
        if save_definition:
            
            self.save_configuration(save_exposed_values=save_definition)
        
    @final
    def load_state(self): 
        
        '''
        Loads the state of a component from a folder path
        If the folder path is not provided, it will use the artifact directory of the component
        
        This assumes that the input, exposed values, and so on are already correctly set in the component
        This method should not typically be called directly, but rather through the load_component_with_state_from_folder function
        '''
        
        folder_path = self.get_artifact_directory()
            
        if not os.path.exists(folder_path):
            raise Exception(f"Folder path {folder_path} does not exist, cannot load state of component")
        
    
    
    
def __load_state_recursive_child_components(origin_component : Component):
        
    '''Loads state of child components recursively'''
    
    for child_component in origin_component.child_components:
        
        __load_state_recursive_child_components(child_component)
        
        if isinstance(child_component, StatefulComponent): # for all the stateful components, we load their state
            
            child_component.load_state()
            
    
    if isinstance(origin_component, StatefulComponent):
        origin_component.load_state(recursive=False)
            
                
def load_component_with_state_from_folder(folder_path) -> Component: #note this is not a method 
    
    '''Loads the state of a component from a folder path'''
    
    json_str = read_text_from_file(folder_path, f'{CONFIGURATION_FILE_NAME}.json')
    
    component_to_return = gen_component_from(json_str)
    
    if not isinstance(component_to_return, ArtifactComponent):
        raise Exception("Tried to load state of component which is not a ArtifactComponent component")
    
    return component_to_return




def save_component_with_state_to_folder(component : ArtifactComponent, folder_path, save_definition=True) -> None:

        '''
        Saves the state of this component and child components        
        '''
        
        for child_component in component.child_components:
                        
            save_component_with_state_to_folder(child_component, False)
            
            if isinstance(child_component, StatefulComponent):

                child_component.save_state(False)
                
                
        if isinstance(component, StatefulComponent):
            component.save_state(save_definition)
    


class StatefulComponentLoader(ArtifactComponent):
    
    '''A component with the capability of storing its state in its respective directory and later load it'''
    
    def define_component_to_save_load(self, component : ArtifactComponent):
        
        if not isinstance(component, ArtifactComponent):
            raise Exception(f"Tried to define component to save / load that is not a ArtifactComponent, but a {type(component)}")
        
        
        
        self.component_to_save_load = component
        self.component_to_save_type = type(component)
        

        
    def proccess_input(self):
        
        super().proccess_input()    
        
        self.pass_input({"artifact_relative_directory" : str(self.component_to_save_load.input["artifact_relative_directory"])}) #str is used to clone the string
        self.pass_input({"base_directory" : str(self.component_to_save_load.input["base_directory"])})
        
        
    @requires_input_proccess
    def save_component(self):
        save_component_with_state_to_folder(self.component_to_save_load, self.component_to_save_load.get_artifact_directory(), save_definition=True)
    
    @requires_input_proccess
    def unload_component(self):
        
        weak_ref = weakref.ref(self.component_to_save_load)

        del self.component_to_save_load

        gc.collect()

        if weak_ref() is not None:
            raise Exception("Component was not fully unloaded — still referenced elsewhere.")
        
    @requires_input_proccess
    def save_and_onload_component(self):
        
        self.save_component()
        self.unload_component()

        
    @requires_input_proccess
    def get_component(self):
        '''Gets the component, if not loaded yet, it is loaded'''
        
        if not hasattr(self, 'component_to_save_load'):
            self.load_component() 
        
        return self.component_to_save_load
    
    
    
    @requires_input_proccess
    def load_component(self):
        '''
        Loads the component from the artifact directory and returns it
        '''
        
        if hasattr(self, 'component_to_save_load'):
            raise Exception("Component is already loaded, cannot load it again")
        
        self.component_to_save_load = load_component_with_state_from_folder(self.artifact_directory)
        
        return self.component_to_save_load