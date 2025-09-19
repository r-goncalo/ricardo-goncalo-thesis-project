


import shutil
from typing import final
from automl.component import Component, requires_input_proccess
from automl.utils.json_component_utils import decode_components_from_dict, dict_from_json_string, gen_component_from, json_string_of_component, component_from_json_string, set_values_of_dict_in_component
from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.basic_components.artifact_management import ArtifactComponent, find_artifact_component_first_parent_directory
import os

from automl.consts import CONFIGURATION_FILE_NAME

import weakref
import gc

import torch
                
                
                
                

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
        This method should not typically be called directly, as it does not save the state of child components
        '''
            
        if save_definition:
            
            self.save_configuration(save_exposed_values=True)
            
        self._save_state_internal()

        
    def _save_state_internal(self):
        pass
        
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
        
        self._load_state_internal()
        

    def _change_to_new_artifact_directory_internal(self, new_folder_path):
        
        super()._change_to_new_artifact_directory_internal(new_folder_path)

        current_artifact_directory = self.get_artifact_directory()

        # copy all files (not folders) from current_artifact_directory to new_folder_path
        for item in os.listdir(current_artifact_directory):
        
            src_path = os.path.join(current_artifact_directory, item)
            dst_path = os.path.join(new_folder_path, item)
            if os.path.isfile(src_path):  # copy only files
                shutil.copy2(src_path, dst_path)
    


    def _load_state_internal(self):
        pass    
    

    def on_unload(self):
        '''
        This method is called when the component is unloaded, it should be used to clean up resources
        '''
        pass
    
    
def __load_state_recursive_child_components(origin_component : Component):
        
    '''Loads state of child components recursively'''
    
    for child_component in origin_component.child_components:
        
        __load_state_recursive_child_components(child_component)
        
        if isinstance(child_component, StatefulComponent): # for all the stateful components, we load their state
            
            child_component.load_state()
            
    
    if isinstance(origin_component, StatefulComponent):
        origin_component.load_state()

# TODO: Complete this
def __change_artifact_directory_recursive_child_components(origin_component : ArtifactComponent):
    pass


def change_artifact_directory_and_child_components(origin_component : ArtifactComponent, new_folder_path):

    origin_component.change_to_new_artifact_directory(new_folder_path)

    __change_artifact_directory_recursive_child_components(origin_component)



def load_component_from_folder(folder_path, configuration_file=CONFIGURATION_FILE_NAME, new_folder_path=None, parent_component_to_be : Component =None) -> Component: #note this is not a method

    '''Loads the state of a component from a folder path'''
    
    json_str = read_text_from_file(folder_path, configuration_file)

    old_folder_last_directory = os.path.basename(folder_path)
    
    component_to_return = gen_component_from(json_str)
    
    if not isinstance(component_to_return, ArtifactComponent):
        print("WARNING: Tried to load state of component which is not a ArtifactComponent component")
        #raise Exception("Tried to load state of component which is not a ArtifactComponent component")

    else: # is instance of ArtifactComponent

        component_to_return.pass_input({"artifact_relative_directory" : '',
                                        "base_directory" : folder_path,
                                        "create_new_directory" : False})
            
        
        if isinstance(component_to_return, StatefulComponent):

            if new_folder_path is not None:

                change_artifact_directory_and_child_components(component_to_return, new_folder_path)

            # if there is a parent component and no defined new folder path
            elif new_folder_path is None and parent_component_to_be is not None:

                first_parent_artifact_component_directory = find_artifact_component_first_parent_directory(parent_component_to_be)

                if first_parent_artifact_component_directory is not None:

                    change_artifact_directory_and_child_components(component_to_return, os.path.join(first_parent_artifact_component_directory, old_folder_last_directory))



            component_to_return.load_state()

            __load_state_recursive_child_components(component_to_return) # load state of children

    
    return component_to_return


def save_state(component : Component, save_definition=True) -> None:

    '''
    Saves the state of this component and child components        
    '''
    
    for child_component in component.child_components:
        
        save_state(child_component)
        
        if isinstance(child_component, StatefulComponent):
            child_component.save_state(save_definition=False)

    if isinstance(component, StatefulComponent):
        component.save_state(save_definition=save_definition)

    elif isinstance(component, ArtifactComponent):
        if save_definition:
            component.save_configuration(True)
        

# TODO: This is wrong
def save_component_with_state_to_folder(component : ArtifactComponent, folder_path, save_definition=True) -> None:
        pass
            

def unload_component(component : Component) -> None:

    '''
    Saves the state of this component and child components        
    '''
    
    for child_component in component.child_components:
        
        unload_component(child_component)
        
        if isinstance(child_component, StatefulComponent):
            child_component.on_unload()
            
    if isinstance(component, StatefulComponent):
        component.on_unload()
                
        
                        
            
                
                
    


class StatefulComponentLoader(ArtifactComponent):
    
    '''A component with the capability of storing its state in its respective directory and later load it'''
    
    def define_component_to_save_load(self, component : ArtifactComponent):
        
        '''
        Defines an Artifact Component as the component to be saved and loaded
        That component has to already have its artifact directory generated
        '''
        
        if not isinstance(component, ArtifactComponent):
            raise Exception(f"Tried to define component to save / load that is not a ArtifactComponent, but a {type(component)}")
        
        self.component_to_save_load = component
        self.component_to_save_type = type(component)
        
        self.input["artifact_relative_directory"] = ''
        self.input["base_directory"] = component.get_artifact_directory()
        

        
    def proccess_input_internal(self):
        
        super().proccess_input_internal()    
        
        if not hasattr(self, 'component_to_save_load'):
            raise Exception("Component to save / load was not defined, use define_component_to_save_load method")
        
        self.pass_input({"artifact_relative_directory" : str(self.component_to_save_load.input["artifact_relative_directory"])}) #str is used to clone the string
        self.pass_input({"base_directory" : str(self.component_to_save_load.input["base_directory"])})
        
        
    @requires_input_proccess
    def save_component(self):
        '''Saves component to its folder'''
        save_state(self.component_to_save_load, save_definition=True)
    
    @requires_input_proccess
    def unload_component(self):

        '''Unloads component, note that it does not implicitly save it first'''
                
        weak_ref = weakref.ref(self.component_to_save_load)

        del self.component_to_save_load

        gc.collect()

        if weak_ref() is not None:
            raise Exception("Component was not fully unloaded — still referenced elsewhere.")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Get memory before
            before = torch.cuda.memory_allocated(device)
            
            print(f"Memory allocated before freeing: {before} bytes, {before / (1024 * 1024)} MB")
    
            # Clean memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Optional: Collect unused IPC memor
            
            after = torch.cuda.memory_allocated(device)
            
            print(f"Memory allocated freed: {before - after} bytes, {(before - after) / (1024 * 1024)} MB")
            
        
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
        
        self.component_to_save_load = load_component_from_folder(self.get_artifact_directory())
        
        return self.component_to_save_load