
from automl.component import Component, requires_input_proccess
from automl.utils.json_component_utils import json_string_of_component, component_from_json_string
from automl.utils.files_utils import open_or_create_folder, write_text_to_file
from automl.core.input_management import InputSignature
from automl.consts import CONFIGURATION_FILE_NAME

import os

def on_artifact_directory_change(self : Component):
        
    if all(key in self.input.keys() for key in ["artifact_relative_directory", "base_directory", "create_new_directory"]): #if there is enough to create an artifact directory   

        
        if hasattr(self, "artifact_directory"):
            print(f"Warning: Artifact directory was changed after it already being generated: {self.artifact_directory}, will not automatically change it again")
            print(f"Warning: Now base directory is {self.base_directory} and relative directory is {self.artifact_relative_directory}")
        
        #else:
        #    print(f"Generating artifact directory due to change in input, instead of inpput procesing")
        #    self.get_artifact_directory()
        
        


class ArtifactComponent(Component):

    '''A component which has a directory associated with it'''
    
    
    parameters_signature = {
        
                        "create_new_directory" : InputSignature(
                            priority=1, default_value=True, ignore_at_serialization=True,
                            description="If it is supposed to create a new directory if existent"),
        
                        "artifact_relative_directory" : InputSignature(
                                priority=2,
                                generator= lambda self : self.name, #the default value for the name of the artifact folder is its name
                                on_pass=on_artifact_directory_change
                                ),
                        
                        "base_directory" : InputSignature(
                            priority=2,
                            ignore_at_serialization=True,
                            default_value='',
                            on_pass=on_artifact_directory_change,
                            description='This path is used as basis to calculate the artifact directory of a component'
                        )
                       }
  
            
    def on_parent_component_defined(self):
        '''Artifact Components try to get the directory of a parent component to use as a base directory'''
        super().on_parent_component_defined()
        
        current_parent_component = self.parent_component
        
        while current_parent_component != None: #looks for a parent component which is an Artifact Component and sets its directory based on it
            
            if isinstance(current_parent_component, ArtifactComponent):
                self.pass_input({"base_directory" : current_parent_component.get_artifact_directory()}) 
                break
            
            current_parent_component = current_parent_component.parent_component
            
    
    def __generate_artifact_directory(self):
        
        if not all(key in self.input.keys() for key in ["artifact_relative_directory", "base_directory", "create_new_directory"]):
            raise Exception(f"Artifact {self.name} with type {type(self)} trying to create a directory without the necessary parameters")
        
        self.artifact_relative_directory = self.input["artifact_relative_directory"]
        
        self.base_directory = self.input["base_directory"]
        
        if self.base_directory == '' and self.artifact_relative_directory == '':
            raise Exception("No path specification for artifact directory")
        
        full_path = os.path.join(self.base_directory, self.artifact_relative_directory)
        
        try:
            self.artifact_directory = open_or_create_folder(full_path, create_new=self.input["create_new_directory"])
            
        except Exception as e:
            
            raise Exception(f"Could not open or create folder with base directory \'{self.base_directory}\' and artifact relative directory \'{self.artifact_relative_directory}\', full directory {full_path}")
                
    
    def generate_artifact_directory(self):
        
        '''Forces generation of artifact directory for this component and raises an exception if it already has it generated'''
        
        if hasattr(self, "artifact_directory"):
            raise Exception(f"Component {self.name} already had an artifact directory generated, \'{self.artifact_directory}\'")
            
        self.__generate_artifact_directory()
        
    
    def get_artifact_directory(self):
        '''Gets (and sets if needed) the artifact directory'''      
        
        if not hasattr(self, "artifact_directory"):
            self.__generate_artifact_directory()
            
        return self.artifact_directory
  
        
    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super().proccess_input_internal()
        
        self.get_artifact_directory()
        
    
    def save_configuration(self, save_exposed_values=False):
        
        json_str = json_string_of_component(self, save_exposed_values=save_exposed_values)
        
        write_text_to_file(self.artifact_directory, f'{CONFIGURATION_FILE_NAME}.json', json_str)  