from datetime import datetime
import pickle
from automl.component import InputSignature, Component, requires_input_proccess

from automl.utils.files_utils import  saveDataframe

from enum import Enum

from automl.basic_components.artifact_management import ArtifactComponent

import os

import pandas as pd

from automl.utils.smart_enum import SmartEnum


class DEBUG_LEVEL(SmartEnum):
    NONE = -1
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    DEBUG = 3
    INFO = 4


IDENT_SPACE = '    '

# LOGGING SCHEMA  -------------------------------------------------------------------------------------------------   


class LoggerSchema(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''

    
    
    parameters_signature = {

                       "necessary_logger_level" : InputSignature(
                            #default_value=DEBUG_LEVEL.INFO, 
                            default_value=DEBUG_LEVEL.ERROR,
                            ignore_at_serialization=True),

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

    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()
            
        self.necessary_logger_level = DEBUG_LEVEL.from_value(self.get_input_value("necessary_logger_level"))  
        
        self.default_print = self.get_input_value("default_print")
        
        self.default_use_timestamp = self.get_input_value("user_timestamp_in_logs")
        
        self.default_log_text_file = self.get_input_value("log_text_file")
        
        self.object_with_name = self.get_input_value("object_with_name") if "object_with_name" in self.input.keys() else None

    # LOGGING -----------------------------------------------------------------------------        

    @requires_input_proccess
    def writeLine(self, string : str = "", file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
    
        '''
        This writes a line to the logger of the component, meant to be called by outside of the scope of the component
        '''

        return self._writeLine(string, file, level, toPrint, use_time_stamp, str_before, ident_level)
    
    @requires_input_proccess
    def change_logger_level(self, new_level : DEBUG_LEVEL):

        self.necessary_logger_level = new_level
    
            
    def _writeLine(self, string : str, file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
        
        '''
        This writes a line to the logger of the component, meant to be called inside the scope of the component
        It does not require the component to have its input processed
        '''
        
        if self.necessary_logger_level.value >= level.value: #if the level of the message is lower than the default level, we write it (more important than what was asked)

            if toPrint == None:
                toPrint = self.default_print
            
            if use_time_stamp == None: #if it was not specified, we use the default value
                use_time_stamp = self.default_use_timestamp
                
            if ident_level > 0:
                for _ in range(ident_level):
                    string = IDENT_SPACE + string

            if use_time_stamp: #if we want to use the timestamp, we add it to the string
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                string = f'[{timestamp}] {string}'
                
            string = f"{str_before}{string}"

            if self.object_with_name is not None: #if we have an object with a name, we add it to the string
                string = f'{self.object_with_name.name}: {string}'

            if file is None:
                file = self.default_log_text_file
            
            string = f'{string}'
            
            if toPrint:
                print(string)

            fd = open(os.path.join(self.get_artifact_directory(), file), 'a')
            fd.write(f'{string}\n')
            fd.close()
        
        
    # note this does not necessary needs all input to be processed
    def saveFile(self, data, directory='', filename='data'): #saves a file using the directory of this log object as a point of reference
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)
        
        fd = open(os.path.join(self.get_artifact_directory(), directory, filename), 'wb') 
        pickle.dump(data, fd)
        fd.close()
    
    # note this does not necessary needs all input to be processed
    def saveDataframe(self, df, directory='', filename='dataframe.csv'): #saves a dataframe using this log object as a reference
        
        '''
        Saves dataframe in artifact directory
        This does not trigger input processing
        '''
        
        if(directory != ''):
            self.createDirIfNotExistent(directory)

        saveDataframe(df, os.path.join(self.get_artifact_directory(), directory), filename)

    def loadDataframe(self, directory='', filename='dataframe.csv'):
        '''
        Loads dataframe in artifact directory
        This does not trigger input processing
        '''
        return pd.read_csv(os.path.join(self.get_artifact_directory(), directory, filename))
    
    @requires_input_proccess
    def openFile(self, *args, **kargs):
        return self.lg.openFile(*args, **kargs)

                
    def createDirIfNotExistent(self, dir): #creates a dir if it does no exist
        
        dir = os.path.join(self.get_artifact_directory(), dir)
        
        try:
            os.listdir(dir)
        
        except:
            os.makedirs(dir)
            
    @requires_input_proccess
    def openFile(self, fileRelativePath): #reads and returns a file
        fd = open(os.path.join(self.get_artifact_directory(), fileRelativePath), 'rb')
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
            
    self.lg  = self.get_input_value("logger_object")
    
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


    def _proccess_input_internal(self): #this is the best method to have initialization done right after
            
        super()._proccess_input_internal()
        
        self.lg : LoggerSchema = self.get_input_value("logger_object") if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
        

    @requires_input_proccess
    def change_logger_level(self, new_level : DEBUG_LEVEL):
        self.lg.change_logger_level(new_level)

    