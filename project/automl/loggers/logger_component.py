from datetime import datetime
import pickle
from automl.component import InputSignature, Component, requires_input_proccess

from automl.utils.json_component_utils import json_string_of_component, component_from_json_string

from automl.utils.files_utils import open_or_create_folder

from enum import Enum

from automl.basic_components.artifact_management import ArtifactComponent

import os

import pandas as pd


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

                       "default_print" : InputSignature(default_value=False, ignore_at_serialization=True),

                       "artifact_relative_directory" : InputSignature(
                                priority=1,
                                default_value="log"
                                ),
                                              
                       "user_timestamp_in_logs" : InputSignature(default_value=True, ignore_at_serialization=True),
                       
                       "log_text_file" : InputSignature(default_value='log.txt', ignore_at_serialization=True, description="The name of the log text file, if it is not specified, it will be created as 'log.txt' in the log directory"),
                    
                        "object_with_name" : InputSignature(mandatory=False, ignore_at_serialization=True, description="The object that will be used to create the profile for the logger, if it is not specified, the logger will not have a profile"),
                    
                    }
        

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
            
        self.default_logger_level = self.input["logger_level"]  
        
        self.default_print = self.input["default_print"]
        
        self.default_use_timestamp = self.input["user_timestamp_in_logs"]
        
        self.default_log_text_file = self.input["log_text_file"]
        
        self.object_with_name = self.input["object_with_name"] if "object_with_name" in self.input.keys() else None

    # LOGGING -----------------------------------------------------------------------------        

    @requires_input_proccess
    def writeLine(self, string : str, file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, **kargs):
        
        
        if self.default_logger_level.value <= level.value: #if the level of the message is lower than the default level, we write it (more important than what was asked)

            if toPrint == None:
                toPrint = self.default_print
                
            if self.object_with_name is not None: #if we have an object with a name, we add it to the string
                string = f'{self.object_with_name.name}: {string}'
            
            if use_time_stamp == None: #if it was not specified, we use the default value
                use_time_stamp = self.input["user_timestamp_in_logs"]
                
            if use_time_stamp: #if we want to use the timestamp, we add it to the string
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                string = f'[{timestamp}] {string}'
                

            if file is None:
                file = self.default_log_text_file
            
            string = f'{string}'
            
            if toPrint:
                print(string)

            fd = open(os.path.join(self.artifact_directory, file), 'a')
            fd.write(f'{string}\n')
            fd.close()
        
        
        
    @requires_input_proccess
    def saveFile(self, data, directory='', filename='data'): #saves a file using the directory of this log object as a point of reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)
        
        fd = open(os.path.join(self.artifact_directory, directory, filename), 'wb') 
        pickle.dump(data, fd)
        fd.close()
    
    @requires_input_proccess
    def saveDataframe(self, df, directory='', filename='dataframe.csv'): #saves a dataframe using this log object as a reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)

        df.to_csv(os.path.join(self.artifact_directory, directory, filename), index=False)

    @requires_input_proccess
    def loadDataframe(self, directory='', filename='dataframe.csv'):
        return pd.read_csv(os.path.join(self.artifact_directory, directory, filename))
    
    @requires_input_proccess
    def openFile(self, *args, **kargs):
        return self.lg.openFile(*args, **kargs)

                
    def createDirIfNotExistent(self, dir): #creates a dir if it does no exist
        
        dir = os.path.join(self.artifact_directory, dir)
        
        try:
            os.listdir(dir)
        
        except:
            os.makedirs(dir)
            
    @requires_input_proccess
    def openFile(self, fileRelativePath): #reads and returns a file
        fd = open(os.path.join(self.artifact_directory, fileRelativePath), 'rb')
        toReturn = pickle.load(fd)
        fd.close()
        return toReturn
    
    @requires_input_proccess
    def createProfile(self, name : str = '', object_with_name = None):            
        
        '''Creates a new logger object with the only difference of this one being the used name'''
        
        input_of_copy = {**self.input} #copy the input to avoid modifying the original one
        
        if object_with_name is not None:
            input_of_copy['object_with_name'] = object_with_name
        
        elif name != '':
            object_with_name = object() #create simple object with a name attribute
            object_with_name.name = name
            input_of_copy["object_with_name"] = object_with_name
        else:
            raise ValueError("You must provide a name or an object with a name to create a profile")

        return LoggerSchema(input=input_of_copy)
    
    
    
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
        
        self.lg : LoggerSchema = self.input["logger_object"] if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
        
        


    