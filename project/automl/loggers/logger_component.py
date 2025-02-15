from automl.component import InputSignature, Schema, requires_input_proccess

from logger.Log import LogClass, openLog

from automl.utils.json_component_utils import json_string_of_component, component_from_json_string

from automl.utils.files_utils import open_or_create_folder


def on_log_pass(self : Schema):
    
    self.lg = self.input["logger_object"]
    

def generate_log_object(self : Schema):
    
    directory = self.input["logger_directory"]
    
    return openLog(logDir=directory, useLogName=False)


def generate_log_directory(self : Schema):
    
    if "logger_object" in self.input.keys():
        lg_object : LogClass = self.lg
        return lg_object.logDir
    
    else:
        return open_or_create_folder('data\\experiments', folder_name=self.name, create_new=self.input["create_directory_if_existent"])


class LoggerSchema(Schema):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    parameters_signature = {
        
                        "create_directory_if_existent" : InputSignature(priority=3, default_value=True),
        
                        "logger_directory" : InputSignature(
                                priority=5,
                                generator=lambda self : generate_log_directory(self)
                                ),
                        
                       "logger_object" : InputSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = lambda self : generate_log_object(self), on_pass=on_log_pass),
                       
                       "create_profile_for_parent" : InputSignature(default_value=False),
                       "create_profile_for_logger" : InputSignature(default_value=True)
                       }
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.lg : LogClass = self.input["logger_object"] if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
    
        if self.input["create_profile_for_parent"]:
            self.lg = self.lg.createProfile(object_with_name=self.parent_component)
    
        elif self.input["create_profile_for_logger"]:
            self.lg = self.lg.createProfile(object_with_name=self)
            
            
    # LOGGING -----------------------------------------------------------------------------        
            
    def writeToFile(self, *args, **kargs):
        return self.lg.writeToFile(*args, **kargs)
                
    def writeLine(self, *args, **kargs):
        return self.lg.writeLine(*args, **kargs)
        
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
        
    def save_configuration(self, parent_component, toPrint=False):
        
        json_string = json_string_of_component(parent_component)
        
        self.lg.writeToFile(string=json_string, file='configuration.json', toPrint=toPrint)
    
    def load_configuration(path):
        
        fd = open(path, 'r') 
        json_string = fd.read()
        fd.close()
        
        return  component_from_json_string(json_string)
    