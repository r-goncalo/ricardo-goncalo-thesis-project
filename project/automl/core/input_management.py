
from enum import Enum

from automl.utils.smart_enum import SmartEnum

from automl.consts import ONLY_IGNORE_AT_SERIALIZATION_AFTER_GET

DEFAULT_GET_FROM_PARENT = False
DEFAULT_IGNORE_AT_SERIALIZATION = False
DEFAULT_PRIORITY = 50
DEFAULT_MANDATORY = True

class InputSignature():
    
    
    def __init__(self,
                 get_from_parent = None, 
                 default_value = None, 
                 generator = None, 
                 validity_verificator=[], 
                 possible_types =[], 
                 description=None, 
                 ignore_at_serialization=None, 
                 priority=None, 
                 on_pass=[], 
                 mandatory=None):
        
        self.default_value = default_value
        self.generator = generator
        
        self.possible_types = possible_types if isinstance(possible_types, list) else [possible_types]
        
        self.description = description
        self.ignore_at_serialization = ignore_at_serialization
        self.priority = priority
        
        self.mandatory = mandatory
        self.get_from_parent = get_from_parent

        self.on_pass = on_pass if isinstance(on_pass, list) else [on_pass]
        self.validity_verificator = validity_verificator if isinstance(validity_verificator, list) else [validity_verificator]

    
    def setup_default_values(self):
        '''
        This sets up the default values for the Input Signature, must be called right before it is ready for use

        The reason this is not done explicitly in __init__ is for the later possibility of delaying setting up default values to after all the fusions are done 
        '''

        if self.get_from_parent == None:
            self.get_from_parent = DEFAULT_GET_FROM_PARENT

        if self.ignore_at_serialization == None:
            self.ignore_at_serialization = DEFAULT_IGNORE_AT_SERIALIZATION

        if self.priority == None:
            self.priority = DEFAULT_PRIORITY

        if self.mandatory == None:
            self.mandatory = DEFAULT_MANDATORY



    def get_value_from_input(self, component_with_input, key, is_none_ok=True):

        '''Gets the value from input, returning None if it does not exist'''

        if is_none_ok:
            to_return = component_with_input.input.get(key, None)
        
        else:
            try:
                to_return = component_with_input.input[key]
            
            except KeyError as e:
                raise Exception(f"Component {component_with_input.name} tried to get mandatory input value for key '{key}' but it was not set in its input") from e


        return to_return

    
    def fuse_with_new(self, other_input_signature):

        '''
        Creates a new InputSignature by fusing this one with another
        The other is treated as new, having some of its values as priority
        '''

        other_input_signature : InputSignature = other_input_signature

        new_default_value = self.default_value if other_input_signature.default_value == None else other_input_signature.default_value

        new_generator = self.generator if other_input_signature.generator == None else other_input_signature.generator

        new_ignore_at_serialization = self.ignore_at_serialization if other_input_signature.ignore_at_serialization == None else other_input_signature.ignore_at_serialization

        new_priority = self.priority if other_input_signature.priority == None else other_input_signature.priority

        new_mandatory = self.mandatory if other_input_signature.mandatory == None else other_input_signature.mandatory

        new_get_from_parent = self.get_from_parent if other_input_signature.get_from_parent == None else other_input_signature.get_from_parent


        new_validity_verificator = [*self.validity_verificator]

        for new_validity_verificator_fun in other_input_signature.validity_verificator:
            if not new_validity_verificator_fun in new_validity_verificator:
                new_validity_verificator.append(new_validity_verificator_fun)
            

        if other_input_signature.possible_types == None:
            new_possible_types = self.possible_types

        else:
            new_possible_types = [*self.possible_types, *other_input_signature.possible_types]
    
        if other_input_signature.description == None:
            new_description = self.description

        elif self.description == None:
            new_description = other_input_signature.description

        else: # if both have description
            new_description = f"{self.description}\n{other_input_signature.description}"


        new_on_pass = [*self.on_pass]

        for new_on_pass_fun in other_input_signature.on_pass:
            if not new_on_pass_fun in new_on_pass:
                new_on_pass.append(new_on_pass_fun)


        new_input_signature = InputSignature(
                 get_from_parent = new_get_from_parent, 
                 default_value = new_default_value, 
                 generator = new_generator, 
                 validity_verificator= new_validity_verificator, 
                 possible_types = new_possible_types, 
                 description= new_description, 
                 ignore_at_serialization= new_ignore_at_serialization, 
                 priority= new_priority, 
                 on_pass= new_on_pass, 
                 mandatory= new_mandatory
        )

        return new_input_signature
    
    def to_dict(self):

        to_dict_to_return = {

            "get_from_parent" : str(self.get_from_parent),
            "default_value" : str(self.default_value),
            "generator" : str(self.generator),
            "validity_verificator" : str(self.validity_verificator),
            "possible_types" : str(self.possible_types),
            "description" : str(self.description),
            "ignore_at_serialization" : str(self.ignore_at_serialization),
            "priority" : str(self.priority),
            "on_pass" : str(self.on_pass),
            "mandatory" : str(self.mandatory),


        }

        return to_dict_to_return




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
    

    def value_was_got(self):
        self.value_got = True
    
    def was_value_got(self):
        return self.value_got
    
    def set_to_ignore_at_serialization(self, value : bool):
        self.ignore_at_serialization = value

