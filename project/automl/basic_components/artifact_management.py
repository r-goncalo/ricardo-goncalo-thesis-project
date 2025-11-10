
from automl.component import Component

from automl.utils.files_utils import open_or_create_folder

from automl.core.input_management import InputSignature

import os

from automl.utils.configuration_component_utils import save_configuration
from automl.loggers.global_logger import globalWriteLine

def on_artifact_directory_change(self : Component):
        
    if self.has_artifact_directory_defined_or_created(): #if there is enough to create an artifact directory   

        
        if hasattr(self, "artifact_directory"):
            globalWriteLine(f"Warning: Artifact directory was changed after it already being generated: {self.artifact_directory}, will not automatically change it again")
            globalWriteLine(f"Warning: Now base directory is {self.base_directory} and relative directory is {self.artifact_relative_directory}")
        
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

        necessary_parameters = ["artifact_relative_directory", "base_directory", "create_new_directory"]
        
        if not all(key in self.input.keys() for key in necessary_parameters):
            raise Exception(f'Artifact {self.name} with type {type(self)} trying to create a directory without the necessary parameters, needs {necessary_parameters} and has {self.input.keys()}')
        
        self.artifact_relative_directory = self.get_input_value("artifact_relative_directory")
        self.base_directory = self.get_input_value("base_directory")
        self.create_new_directory = self.get_input_value("create_new_directory")
        
        if self.base_directory == '' and self.artifact_relative_directory == '':
            raise Exception("No path specification for artifact directory")
        
        try:
            full_path = os.path.join(self.base_directory, self.artifact_relative_directory)
            self.artifact_directory = open_or_create_folder(full_path, create_new=self.create_new_directory)
            
        except Exception as e:
            
            raise Exception(f"Component {self.name} could not open or create folder with base directory \'{self.base_directory}\' and artifact relative directory \'{self.artifact_relative_directory}\', full directory {full_path} due to exception:\n{e}") from e
        
                
    def _force_generate_artifact_directory(self):
        self.__generate_artifact_directory()

    def generate_artifact_directory(self):
        
        '''Forces generation of artifact directory for this component and raises an exception if it already has it generated'''
        
        if hasattr(self, "artifact_directory"):
            raise Exception(f"Component {self.name} already had an artifact directory generated, \'{self.artifact_directory}\'")
            
        self.__generate_artifact_directory()
        
    
    def has_artifact_directory_defined_or_created(self) -> bool:
        return self.has_artifact_directory() or all(key in self.input.keys() for key in ["artifact_relative_directory", "base_directory", "create_new_directory"])

    def has_artifact_directory(self) -> bool:
        return hasattr(self, "artifact_directory")
    
                
    def get_artifact_directory(self):
        '''Gets (and sets if needed) the artifact directory'''      
        
        if not self.has_artifact_directory():
            self.__generate_artifact_directory()
            
        return self.artifact_directory
    
    # CHANGE ARTIFACT DIRECTORY -------------------------------------------------

    def change_to_new_artifact_directory(self, new_folder_path):

        # check if new_folder_path is empty, if not, raise error
        if os.path.exists(new_folder_path) and os.listdir(new_folder_path):
            raise ValueError(f"Target folder '{new_folder_path}' is not empty.")
        
        # create new_folder in path if it does not exist
        os.makedirs(new_folder_path, exist_ok=True)

        self._change_to_new_artifact_directory_internal(new_folder_path)

        self.pass_input({"base_directory" : new_folder_path, 
                         "artifact_relative_directory" : '', 
                         "create_new_directory" : False})

        self._force_generate_artifact_directory()

        self.write_line_to_notes(f"Component had its path changed to {new_folder_path}", use_datetime=True)


    def _change_to_new_artifact_directory_internal(self, new_folder_path):
        '''What happens between the creation of the folder and the actual change of the directory'''
        os.makedirs(new_folder_path, exist_ok=True)


    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()
        
        # self.get_artifact_directory()
        


    def save_configuration(self, save_exposed_values=False, ignore_defaults=True, respect_ignore_order=True):
        
        save_configuration(self, self.get_artifact_directory(), save_exposed_values=save_exposed_values, ignore_defaults=ignore_defaults, respect_ignore_order=respect_ignore_order)



        

def find_artifact_component_first_parent_directory(component : Component):
    
    current_parent_component = component
    
    while current_parent_component != None: #looks for a parent component which is an Artifact Component and sets its directory based on it
        
        if isinstance(current_parent_component, ArtifactComponent):
            return current_parent_component.get_artifact_directory()
        
        current_parent_component = current_parent_component.parent_component
        
    return None
