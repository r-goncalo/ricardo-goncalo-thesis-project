from .component import InputSignature, Component, requires_input_proccess

from logger.Log import LogClass, openLog

class LoggerComponent(Component):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    input_signature = {
                       "logger" : InputSignature(ignore_at_serialization=True, generator= lambda _ : openLog()),
                       
                       "create_profile" : InputSignature(default_value=True, ignore_at_serialization=True)
                       }

    # INITIALIZATION ----------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.lg : LogClass = self.input["logger"]
    
        if self.input["create_profile"]:
            self.lg = self.lg.createProfile(object_with_name=self)
            
        
    def save_configuration(self):
        
        json_string = self.gen_config_json_string()
        
        self.lg.writeToFile(string=json_string, file='configuration.json')
    