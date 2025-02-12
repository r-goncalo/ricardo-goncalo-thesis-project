from .component import InputSignature, Schema, requires_input_proccess

from logger.Log import LogClass, openLog

from .json_component_utils import json_string_of_component, component_from_json_string


def on_log_pass(self):
    
    self.lg = self.input["logger"]
    return self.lg


class LoggerSchema(Schema):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    parameters_signature = {
                       "logger" : InputSignature(ignore_at_serialization=True, priority=0, generator = lambda self : openLog(), on_pass=on_log_pass),
                       
                       "create_profile" : InputSignature(default_value=True, ignore_at_serialization=True)
                       }
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.lg : LogClass = self.input["logger"] if not hasattr(self, "lg") else self.lg #changes self.lg if it does not already exist
    
        if self.input["create_profile"]:
            self.lg = self.lg.createProfile(object_with_name=self)
            
    # CONFIGURATION SAVING / LOADING ------------------------------------------------------    
        
    def save_configuration(self, toPrint=False):
        
        json_string = json_string_of_component(self)
        
        self.lg.writeToFile(string=json_string, file='configuration.json', toPrint=toPrint)
    
    def load_configuration(path):
        
        fd = open(path, 'r') 
        json_string = fd.read()
        fd.close()
        
        return  component_from_json_string(json_string)
    