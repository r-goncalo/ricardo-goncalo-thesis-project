
from enum import Enum

from automl.utils.smart_enum import SmartEnum

from automl.consts import ONLY_IGNORE_AT_SERIALIZATION_AFTER_GET



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


    def get_value_from_input(component_with_input, key, is_none_ok=True):

        '''Gets the value from input, returning None if it does not exist'''

        if is_none_ok:
            to_return = component_with_input.input.get(key, None)
        
        else:
            try:
                to_return = component_with_input.input[key]
            
            except KeyError as e:
                raise Exception(f"Component {component_with_input.name} tried to get mandatory input value for key '{key}' but it was not set in its input") from e


        return to_return


class InputMetaData():
    
    '''
    Represents the metadata of an input of a component
    
    Contains informations such as if it was passed, generated or is a default value
    
    '''
    
    class InputOrigin(SmartEnum):
        
        DEFAULT = 0
        GENERATED = 1
        PASSED = 2   
        
    
    def __init__(self, parameter_signature : InputSignature):
        
        self.origin = InputMetaData.InputOrigin.DEFAULT
        self.parameter_signature : InputSignature = parameter_signature

        self.ignore_at_serialization = parameter_signature.ignore_at_serialization or ONLY_IGNORE_AT_SERIALIZATION_AFTER_GET
    
        self.value_got = False

    def custom_value_passed(self):
        self.origin = InputMetaData.InputOrigin.PASSED
        
    def custom_value_removed(self):
        self.origin = InputMetaData.InputOrigin.DEFAULT
        
    def default_value_was_set(self):
        self.origin = InputMetaData.InputOrigin.DEFAULT
        
    def generator_value_was_set(self):
        self.origin = InputMetaData.InputOrigin.DEFAULT
        
    def was_custom_value_passed(self):
        return self.origin == InputMetaData.InputOrigin.PASSED
    
    def was_value_got(self):
        return self.value_got
    
    def set_to_ignore_at_serialization(self, value : bool):
        self.ignore_at_serialization = value

