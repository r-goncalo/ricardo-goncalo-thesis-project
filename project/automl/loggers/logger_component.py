from automl.component import InputSignature, Schema, requires_input_proccess

from logger.Log import LogClass, openLog

from automl.utils.json_component_utils import json_string_of_component, component_from_json_string

from automl.utils.files_utils import open_or_create_folder

from enum import Enum

BASE_EXPERIMENT_DIRECTORY = 'data\\experiments'


def on_log_pass(self : Schema):
        
    self.lg = self.input["logger_object"]
    self.input["logger_directory"] = self.lg.logDir
    

def generate_log_object(self : Schema):
        
    directory = self.input["logger_directory"]
    
    return openLog(logDir=directory, useLogName=False)


def generate_log_directory(self : Schema):
    
    print(f"generate_log_directory for object {self.name}")
    
    if "logger_object" in self.input.keys():
        lg_object : LogClass = self.lg
        return lg_object.logDir
    
    else:
        return open_or_create_folder(BASE_EXPERIMENT_DIRECTORY, folder_name=self.name, create_new=self.input["create_directory_if_existent"])


class LoggerSchema(Schema):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    class Level(Enum):
        CRITICAL = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        DEBUG = 4

    
    
    parameters_signature = {
        
                        "create_directory_if_existent" : InputSignature(priority=3, default_value=True, ignore_at_serialization=True),
        
                        "logger_directory" : InputSignature(
                                priority=5,
                                generator=lambda self : generate_log_directory(self), 
                                ignore_at_serialization=True
                                ),
                                                
                        "logger_level" : InputSignature(default_value=Level.INFO, ignore_at_serialization=True),
                        
                       "logger_object" : InputSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = lambda self : generate_log_object(self), 
                                                        on_pass=on_log_pass),
                       
                       "create_profile_for_parent" : InputSignature(default_value=False, ignore_at_serialization=True),
                       "create_profile_for_logger" : InputSignature(default_value=True, ignore_at_serialization=True),
                       
                       "default_print" : InputSignature(default_value=False)
                       }
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.lg : LogClass = self.input["logger_object"] if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
    
        self.logger_level : LoggerSchema.Level = self.input["logger_level"]
        
        self.default_print = self.input["default_print"]
    
        if self.input["create_profile_for_parent"]:
            self.lg = self.lg.createProfile(object_with_name=self.parent_component)
    
        elif self.input["create_profile_for_logger"]:
            self.lg = self.lg.createProfile(object_with_name=self)
            
        self.lg.writeLine("Created logger in directory: " + self.lg.logDir)
            
            
    # LOGGING -----------------------------------------------------------------------------        
            
    def writeToFile(self, *args, **kargs):
        return self.lg.writeToFile(*args, **kargs)
                
    def writeLine(self, *args, level=Level.INFO, toPrint=None, **kargs):
        
        if toPrint == None:
            toPrint = self.default_print
        
        if self.logger_level.value <= level.value:
            return self.lg.writeLine(*args, toPrint=toPrint, **kargs)
        
    def saveFile(self, *args, **kargs):
        return self.lg.saveFile(*args, **kargs)
    
    def saveDataframe(self, *args, **kargs):
        return self.lg.saveDataframe(*args, **kargs)
        
    def loadDataframe(self, *args, **kargs):
        return self.lg.loadDataframe(*args, **kargs)
        
    def openFile(self, *args, **kargs):
        return self.lg.openFile(*args, **kargs)
    
    def openChildLog(self, *args, **kargs):
        return self.lg.openChildLog(*args, **kargs)
    
    def createProfile(self, *args, **kargs):
        return self.lg.createProfile(*args, **kargs)
    
            
    # CONFIGURATION SAVING / LOADING ------------------------------------------------------    
        
    def save_configuration(self, toPrint=False):
        
        json_string = json_string_of_component(self)
        
        self.lg.writeToFile(string=json_string, file='configuration.json', toPrint=toPrint)
    
    def load_configuration(path):
        
        fd = open(path, 'r') 
        json_string = fd.read()
        fd.close()
        
        return  component_from_json_string(json_string)
    