
from automl.component import Component, requires_input_proccess
from automl.utils.json_component_utils import json_string_of_component, component_from_json_string
from automl.utils.files_utils import open_or_create_folder
from automl.core.input_management import InputSignature

import os


class ArtifactComponent(Component):

    '''A component which has a directory associated with it'''
    
    parameters_signature = {
        
                        "create_directory" : InputSignature(priority=3, default_value=True, ignore_at_serialization=True),
        
                        "artifact_relative_directory" : InputSignature(
                                priority=5,
                                generator= lambda self : self.name #the default value for the name of the artifact folder is its name
                                ),
                        
                        "base_directory" : InputSignature(
                            priority=4,
                            ignore_at_serialization=True,
                            default_value='',
                            description='This path is used as basis to calculate the artifact directory of a component'
                        )
                       }
  
            
    def on_parent_component_defined(self):
        '''Artifact Components try to get the directory of a parent component to use as a base directory'''
        super().on_parent_component_defined()
        
        current_parent_component = self.parent_component
        
        while current_parent_component != None:
            
            if isinstance(current_parent_component, ArtifactComponent):
                current_parent_component.proccess_input_if_not_proccesd()
                self.pass_input({"base_directory" : current_parent_component.artifact_directory}) 
                break
            
            current_parent_component = current_parent_component.parent_component
            
                
        
    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.artifact_relative_directory = self.input["artifact_relative_directory"]
        
        self.base_directory = self.input["base_directory"]
        
        self.artifact_directory = open_or_create_folder(self.base_directory  + '\\' + self.artifact_relative_directory, folder_name=self.name, create_new=self.input["create_directory"])
        
        print("Created artifact directory at: " + self.artifact_directory)
        