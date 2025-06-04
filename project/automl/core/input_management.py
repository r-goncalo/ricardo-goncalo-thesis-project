
from enum import Enum





class InputSignature():
    
    
    def __init__(self,
                 get_from_parent = False, 
                 default_value = None, 
                 generator = None, 
                 validity_verificator=None, 
                 possible_types : list = [], 
                 description='', 
                 ignore_at_serialization=False, 
                 priority=50, 
                 on_pass=None, 
                 mandatory=True):
        
        self.default_value = default_value
        self.generator = generator
        self.validity_verificator = validity_verificator
        self.possible_types = possible_types
        self.description = description
        self.ignore_at_serialization = ignore_at_serialization
        self.priority = priority
        self.on_pass = on_pass
        self.mandatory = mandatory
        self.get_from_parent = get_from_parent
        
class InputMetaData():
    
    '''
    Represents the metadata of an input of a component
    
    Contains informations such as if it was passed, generated or is a default value
    
    '''
    
    class InputOrigin(Enum):
        
        DEFAULT = 0
        GENERATED = 1
        PASSED = 2   
        
    
    def __init__(self, parameter_signature : InputSignature):
        
        self.origin = InputMetaData.InputOrigin.DEFAULT
        self.parameter_signature = parameter_signature
    
    def custom_value_passed(self):
        self.origin = InputMetaData.InputOrigin.PASSED
        
    def was_custom_value_passed(self):
        return self.origin == InputMetaData.InputOrigin.PASSED
    

