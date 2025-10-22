

from automl.utils.class_util import super_class_from_list_of_class


LOOK_FOR_SUPERCLASSES = True # when this is true, the register will take into account inheritence automatically, without need for explicit configuration


# CUSTOM JSON LOGIC ------------------------------------------------------------------------------------------------------

class CustomJsonLogic():

    '''
    A class meant to be inherited by classes that define custom decoding and encoding logic for themselves or other values
    
    A class that inherits can either implement its methods if its for its own usage or guarantee that the external type to 
    encode / decode saves in its instances itself in the custom_encoding_strategy attribute
    
    '''


    def to_dict(self):

        '''
        The instance is encoded into a dictionary
        
        The dictionary appears in the configuration as {__type__ : custom_logic_type, object : {...}}

        The default method for encoding will also be called recursively for members of the dictionary
        
        '''
        
        raise NotImplementedError()
            
            
    def from_dict(dict, decode_elements_fun, source_component):

        '''
        An instance is generated from the dictionary
        
        The dictionary appears in the configuration as {__type__ : custom_logic_type, object : {...}}
        
        This also receives the base function being used for decoding, in case it is meant to decode sub elements and such
        This also receives de source_component in case the 
        '''

        raise NotImplementedError()
    

#this is meant to register, for certain types, custom json encoding and decoder strategies
__custom_json_strategies_register : dict[type, CustomJsonLogic] = {}

def register_custom_strategy(t : type, custom_logic : CustomJsonLogic):
    print(f"Registing custom strategy with name {custom_logic} for encoding and decoding of objects of type {t}")
    __custom_json_strategies_register[t] = custom_logic

def get_custom_strategy(t : type) -> CustomJsonLogic:

    to_return = __custom_json_strategies_register.get(t, None)

    if to_return == None and LOOK_FOR_SUPERCLASSES: # if we did not find it but we want to check if some registed type is superclass

        super_class = super_class_from_list_of_class(__custom_json_strategies_register.keys(), t)

        if super_class is not None:

            return __custom_json_strategies_register[super_class]
        
    
    return to_return