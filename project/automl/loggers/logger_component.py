from automl.component import InputSignature, Component, requires_input_proccess

from logger.Log import LogClass, openLog

from automl.utils.json_component_utils import json_string_of_component, component_from_json_string

from automl.utils.files_utils import open_or_create_folder

from enum import Enum

from automl.basic_components.artifact_management import ArtifactComponent


class DEBUG_LEVEL(Enum):
    NONE = -1
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    DEBUG = 3
    INFO = 4


# LOGGING SCHEMA  -------------------------------------------------------------------------------------------------   


class LoggerSchema(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''

    
    
    parameters_signature = {
        
                                                
                        "logger_level" : InputSignature(default_value=DEBUG_LEVEL.INFO, ignore_at_serialization=True),
                       
                       "create_profile_for_parent" : InputSignature(default_value=True, ignore_at_serialization=True, description="if the entity responsible for the messages is the parent of the logger object"),
                       "create_profile_for_logger" : InputSignature(default_value=False, ignore_at_serialization=True, description="If the entity responsible for the messages is the logger object"),
                       
                       "default_print" : InputSignature(default_value=False, ignore_at_serialization=True),
                       
                       "artifact_relative_directory" : InputSignature(
                                priority=1,
                                default_value="log"
                                ),
                       
                       "logger_object" : InputSignature(mandatory=False, ignore_at_serialization=True)
                       }
        

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        if "logger_object" in self.input.keys():
            self.lg = self.input["logger_object"]
            
        else:
            self.lg : LogClass = openLog(logDir=self.artifact_directory, useLogName=False)
            
        if self.input["create_profile_for_parent"]:
            self.lg = self.lg.createProfile(object_with_name=self.parent_component)
    
        elif self.input["create_profile_for_logger"]:
            self.lg = self.lg.createProfile(object_with_name=self)  
            
        self.logger_level = self.input["logger_level"]  
        
        self.default_print = self.input["default_print"]
                        
            
    # LOGGING -----------------------------------------------------------------------------        
        
    @requires_input_proccess
    def writeToFile(self, *args, **kargs):
        return self.lg.writeToFile(*args, **kargs)
                
    @requires_input_proccess
    def writeLine(self, *args, level=DEBUG_LEVEL.INFO, toPrint=None, **kargs):
        
        if toPrint == None:
            toPrint = self.default_print
        
        if self.logger_level.value <= level.value:
            return self.lg.writeLine(*args, toPrint=toPrint, **kargs)
        
        
        
    @requires_input_proccess
    def saveFile(self, *args, **kargs):
        return self.lg.saveFile(*args, **kargs)
    
    @requires_input_proccess
    def saveDataframe(self, *args, **kargs):
        return self.lg.saveDataframe(*args, **kargs)
        
    @requires_input_proccess
    def loadDataframe(self, *args, **kargs):
        return self.lg.loadDataframe(*args, **kargs)
        
    @requires_input_proccess
    def openFile(self, *args, **kargs):
        return self.lg.openFile(*args, **kargs)
    
    @requires_input_proccess
    def openChildLog(self, *args, **kargs):
        return self.lg.openChildLog(*args, **kargs)
    
    @requires_input_proccess
    def createProfile(self, *args, **kargs):
        
        return LoggerSchema(input={**self.input, "logger_object" : self.lg.createProfile(*args, **kargs)})
    
    
# COMPONENT WITH LOGGING -------------------------------------------------------------------------------------------------    
    
def on_log_pass(self : Component):
            
    self.lg  = self.input["logger_object"]
    
def generate_logger_for_component(self : ArtifactComponent):
    return self.initialize_child_component(LoggerSchema, input={
            "create_new_directory" : False,
            "base_directory" : self.get_artifact_directory(), 
            "artifact_relative_directory" : ""}
        )

# TODO: In the future, components may extend this, but not the LoggingComponent
class ComponentWithLogging(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    
    parameters_signature = {
                                                                                
                       "logger_object" : InputSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = generate_logger_for_component , 
                                                        on_pass=on_log_pass)
                       }


    def proccess_input(self): #this is the best method to have initialization done right after
            
        super().proccess_input()
        
        self.lg : LogClass = self.input["logger_object"] if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
        
        


    