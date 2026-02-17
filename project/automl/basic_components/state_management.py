


import shutil
import subprocess
from typing import final
from automl.component import Component, requires_input_proccess
from automl.utils.json_utils.json_component_utils import  gen_component_from
from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.basic_components.artifact_management import ArtifactComponent, find_artifact_component_first_parent_directory
import os

from automl.consts import CONFIGURATION_FILE_NAME

import weakref
import gc
import inspect

from automl.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automl.loggers.global_logger import globalWriteLine
from automl.loggers.logger_component import flush_text_of_all_loggers_and_children
import torch
                
import sys

from automl.core.global_class_registry import get_registered_classes_generators, has_registered_classes_generators, serialize_registered_classes, load_custom_classes
                
                

class StatefulComponent(ArtifactComponent):
    
    '''
    A component with the capability of storing its state in its respective directory and later load it
    All classes which extend this class should implement the save_state method and load_state function
    
    Note that all components can inherently store and load their state if it is purely their input, output and exposed values
    '''
    
    @final
    def save_state(self, save_definition=True):
        '''
        Saves the state of this component        
        This method should not typically be called directly, as it does not save the state of child components
        '''
            
        if save_definition:
            self.save_configuration(save_exposed_values=True, ignore_defaults=False, respect_ignore_order=False)
            
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
    

    def _on_unload(self):
        '''
        This method is called when the component is unloaded, it should be used to clean up resources
        '''
        pass
    
    
def __load_state_recursive_child_components(origin_component : Component):
        
    '''Loads state of child components recursively'''

    origin_component.setup_default_value_if_no_value("artifact_relative_directory")
    origin_component.pass_input({"create_new_directory" : False})
    
    for child_component in origin_component.child_components:
        __load_state_recursive_child_components(child_component)
    
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
    
    registered_classes_folder = os.path.join(folder_path, "__custom_classes", "custom_classes.py")

    has_registered_classes_in_folder = os.path.exists(registered_classes_folder)

    globalWriteLine(f"Has registered custom classes in {registered_classes_folder}: {has_registered_classes_in_folder}", file="global_classes.txt")

    if has_registered_classes_in_folder:
        load_custom_classes(registered_classes_folder)

    old_folder_last_directory = os.path.basename(folder_path)
    
    component_to_return = gen_component_from(json_str)

    component_to_return.write_line_to_notes(f"Component generated from folder {folder_path}", use_datetime=True)
    
    if not isinstance(component_to_return, ArtifactComponent):
        globalWriteLine(f"WARNING: Tried to load state of component which is not a ArtifactComponent component: {component_to_return}")
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
    The state is first saved on child components, and later in the component that initiated the method        
    '''
    
    for child_component in component.child_components:
        
        save_state(child_component, save_definition=False)
        
        if isinstance(child_component, StatefulComponent):
            child_component.save_state(save_definition=False)

    if isinstance(component, StatefulComponent):
        component.save_state(save_definition=save_definition)

    elif isinstance(component, ArtifactComponent):
        if save_definition:
            component.save_configuration(save_exposed_values=True, ignore_defaults=False)

    globalWriteLine(f"Has registered classes: {has_registered_classes_generators()}", file="global_classes.txt")

    if save_definition and has_registered_classes_generators():

        artifact_dir = component.get_artifact_directory()
        custom_dir = os.path.join(artifact_dir, "__custom_classes")
        os.makedirs(custom_dir, exist_ok=True)

        init_file = os.path.join(custom_dir, "__init__.py")
        if not os.path.exists(init_file):
            write_text_to_file(custom_dir, "__init__.py", "")

        if get_registered_classes_generators():
            code = serialize_registered_classes()
            write_text_to_file(custom_dir, "custom_classes.py", code)        

# TODO: This is wrong
def save_component_with_state_to_folder(component : ArtifactComponent, folder_path, save_definition=True) -> None:
        pass
            

def unload_component(component : Component) -> None:

    '''
    Unloads this component and child components        
    '''
    
    for child_component in component.child_components:
        
        unload_component(child_component)
        
        if isinstance(child_component, StatefulComponent):
            child_component._on_unload()
            
    if isinstance(component, StatefulComponent):
        component._on_unload()
                
        
                        
            

# TODO: REVIEW THIS, it is weird and unecessarly complex
class StatefulComponentLoader(StatefulComponent):
    
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
        self.input["base_directory"] = str(component.get_artifact_directory())
        self.input["create_new_directory"] = False        

        
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()    
        
        if hasattr(self, 'component_to_save_load'):
        
            self.input["artifact_relative_directory"] = str(self.component_to_save_load.input["artifact_relative_directory"])
            self.input["base_directory"] = str(self.component_to_save_load.input["base_directory"])
            self.input["create_new_directory"] = False   
        

    def _save_state_internal(self):
        super()._save_state_internal()
        self.save_component()

    @requires_input_proccess
    def save_component(self):
        '''Saves component to its folder'''
        save_state(self.component_to_save_load, save_definition=True)

    @requires_input_proccess
    def unload_if_loaded(self):
        if hasattr(self, 'component_to_save_load'):
            self.unload_component() 

    def __deal_with_unwanted_unloaded_component(weak_ref : weakref.ReferenceType):

        leaked_obj = weak_ref()

        referrers = gc.get_referrers(leaked_obj)

        debug_lines = [
            "Component was not fully unloaded, still referenced elsewhere.",
            f"Leaked object id: {id(leaked_obj)}",
            f"Number of referrers: {len(referrers)}",
            "Referrers detail:"
        ]

        for i, ref in enumerate(referrers):
            try:
                ref_type = type(ref)
                location = ""

                # Try to locate where this referrer lives
                if hasattr(ref, "__class__"):
                    location = inspect.getmodule(ref.__class__)

                elif inspect.ismodule(ref):
                    location = ref

                debug_lines.append(
                    f"[{i}] type={ref_type}, repr={repr(ref)[:200]}, module={location}"
                )

            except Exception as e:
                debug_lines.append(f"[{i}] <error inspecting referrer: {e}>")

        raise Exception("\n".join(debug_lines))
    
    @requires_input_proccess
    def unload_component(self):

        '''Unloads component, note that it does not implicitly save it first'''

        # if this or any children components had dataframes, results, or something, we save them
        save_all_dataframes_of_component_and_children(self.component_to_save_load)
        flush_text_of_all_loggers_and_children(self.component_to_save_load)
                
        weak_ref = weakref.ref(self.component_to_save_load)

        del self.component_to_save_load

        gc.collect()

        if weak_ref() is not None:
            self.__deal_with_unwanted_unloaded_component(weak_ref)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Get memory before
            #before = torch.cuda.memory_allocated(device)
            
            #print(f"Memory allocated before freeing: {before} bytes, {before / (1024 * 1024)} MB")
    
            # Clean memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Optional: Collect unused IPC memor
            
            #after = torch.cuda.memory_allocated(device)
            
            #print(f"Memory allocated freed: {before - after} bytes, {(before - after) / (1024 * 1024)} MB")
            
        
    @requires_input_proccess
    def save_and_onload_component(self):
        
        self.save_component()
        self.unload_component()

        
    @requires_input_proccess
    def get_component(self) -> ArtifactComponent:
        '''Gets the component, if not loaded yet, it is loaded'''
        
        if not hasattr(self, 'component_to_save_load'):
            self.load_component() 
        
        return self.component_to_save_load
    
    @requires_input_proccess
    def detach_run_component(self, to_wait = False, global_logger_level = None):
        '''
        If the component is loaded, saves it and unloads it
        It then uses the load_and_run_component command to start a subproccess to run the component
        '''
        artifact_dir = self.get_artifact_directory()

        if hasattr(self, 'component_to_save_load'):
            self.save_component()
            self.unload_component()


        cmd = [
            sys.executable,
            "-m",
            "automl.cli.load_run_component",
            "--component_path",
            artifact_dir,
        ]

        if global_logger_level is not None:
            cmd = [*cmd, "--global_logger_level", global_logger_level]

        popen_proccess = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # fully detached (POSIX)
        )

        if to_wait:
            return_code = popen_proccess.wait()

        return popen_proccess
    
    @requires_input_proccess
    def load_component(self):
        '''
        Loads the component from the artifact directory and returns it
        '''
        
        if hasattr(self, 'component_to_save_load'):
            raise Exception("Component is already loaded, cannot load it again")
        
        self.component_to_save_load = load_component_from_folder(self.get_artifact_directory())
        
        return self.component_to_save_load