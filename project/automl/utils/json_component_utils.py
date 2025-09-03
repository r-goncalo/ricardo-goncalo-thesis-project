import json
import os
from typing import Union

from automl.component import Component, InputSignature, InputMetaData
import pickle

from automl.consts import CONFIGURATION_FILE_NAME, LOADED_COMPONENT_FILE_NAME
from automl.utils.class_util import get_class_from, get_class_from_string

from enum import Enum

from automl.utils.files_utils import write_text_to_file

# ENCODING --------------------------------------------------

class ComponentValuesElementsEncoder(json.JSONEncoder):
    
    '''Encodes elements of a component input or exposed value, which can be a component (defined by its localization) or a primitive type'''
    
    def __init__(self, *args, source_component, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.source_component = source_component
    
    def default(self, obj):
                        
        if isinstance(obj, Component):
            
            if obj.get_source_component() != self.source_component: # TODO: Optimize this
                raise Exception(f"Component {obj.name} is not in the same tree as the source component")

            return {
                "__type__": str(type(obj)),
                "name" : obj.name,
                "localization" : obj.get_index_localization()
            }
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
        
        elif isinstance(obj, Enum):
            return obj.value
        
        elif isinstance(obj, type):
            return str(obj)
        
        elif hasattr(obj, "to_dict"): # if it has a custom to_dict method
            
            if not hasattr(type(obj), "from_dict"):
                raise Exception(f"Object {obj} has a to_dict method, but not a from_dict method in its type {type(obj)}, so it cannot be serialized")

            return {"__type__": str(type(obj)), "object" : self.default(obj.to_dict())}

        try:
            return super().default(obj)
        
        except:
            
            return None
        

class ComponentInputEncoder(json.JSONEncoder):
    
    '''Used with component to encode its input (not the input itself)'''
    
    def __init__(self, *args, ignore_defaults : bool, source_component, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
        self.source_component = source_component
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            input = obj.input
            
            toReturn = {}
            
            for key in input.keys():
                
                parameter_meta : InputMetaData = obj.get_input_meta()[key]
                parameters_signature : InputSignature = parameter_meta.parameter_signature
                                
                if ( not parameters_signature.ignore_at_serialization ) and (not ( ( not parameter_meta.was_custom_value_passed() ) and self.ignore_defaults )):
                    
                    serialized_value = json.dumps(input[key], cls=ComponentValuesElementsEncoder, source_component=self.source_component) #for each value in input, loads
                    
                    if serialized_value != 'null':
                        toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
        
        
class ComponentExposedValuesEncoder(json.JSONEncoder):
    
    '''Used with component to encode its exposed values'''
    
    def __init__(self, *args, source_component, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.source_component = source_component
    
    
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            exposed_values = obj.values
            
            toReturn = {}
            
            for key in exposed_values.keys():
                
                serialized_value = json.dumps(exposed_values[key], cls=ComponentValuesElementsEncoder, source_component=self.source_component) #for each value in exposed_value, loads
                   
                if serialized_value != 'null':
                    toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
            

class ComponentEncoder(json.JSONEncoder):
    
    
    '''Encodes the definition of a component, focusing on its child components'''
    
    def __init__(self, *args, ignore_defaults : bool, save_exposed_values : bool, source_component : Component, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
        self.save_exposed_values = save_exposed_values
        self.source_component = source_component
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            toReturn = {
                "__type__": str(type(obj)),
                "name": obj.name,
                "input": json.loads(json.dumps(obj, cls= ComponentInputEncoder, ignore_defaults=self.ignore_defaults, source_component=self.source_component)),
                }
            
            if self.save_exposed_values:
                
                toReturn["exposed_values"] = json.loads(json.dumps(obj, cls= ComponentExposedValuesEncoder, source_component=self.source_component))

            
            if len(obj.child_components) > 0:
                
                toReturn["child_components"] = self.default(obj.child_components)
                        
            return toReturn
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
        
        
            
        return None

def json_string_of_component_dict(component_dict : dict, ignore_defaults = True, save_exposed_values = False):
    
    component_from_dict = gen_component_from_dict(component_dict)
    
    return json_string_of_component(component_from_dict, ignore_defaults=ignore_defaults, save_exposed_values=save_exposed_values)


# TODO: maybe it should be source_component.get_source_component()?
def json_string_of_component(component, ignore_defaults = False, save_exposed_values = False):
    return json.dumps(component, cls=ComponentEncoder, indent=4, ignore_defaults=ignore_defaults, save_exposed_values = save_exposed_values, source_component=component)

# DECODING --------------------------------------------------------------------------

def decode_a_component_element(source_component : Component, component_element_dict : dict):
    
    keys = component_element_dict.keys()
    
    if "localization" in keys:
            
        component_to_return = source_component.get_child_by_localization(component_element_dict["localization"])
        
    elif "name" in keys: # if it does not have localization and has name
        
        component_to_return = source_component.get_child_by_name(component_element_dict["name"])
        
    if "name" in keys:
        component_to_return.name = component_element_dict["name"]
        
    return component_to_return
    

def decode_a_non_component_element(element_dict : dict):
        
    element_type_str = element_dict["__type__"]
    
    element_type = get_class_from(element_type_str)
    
    if hasattr(element_type, "from_dict"):
        
        if "object" in element_dict.keys():
            
            instanced_object = element_type.from_dict(element_dict["object"])
                        
            return instanced_object
        
        else:
            raise Exception("No object defined when decoding element of type " + element_type_str)
        
    else:
        raise Exception("No from_dict method defined for type " + element_type_str)
    


def decode_components_input_element(source_component : Component, element):
    
    '''
    Decodes an element in a dictionary of a component
    One of the things it does is get a component with a localization and set it
    '''
        
    if isinstance(element, dict):
        keys = element.keys()
        
        if "__type__" in keys:
            
            class_of_element : type = get_class_from(element['__type__'])
            
            if issubclass(class_of_element, Component): #if it is a Schema
                
                return decode_a_component_element(source_component, element)
            
            else:
                return decode_a_non_component_element(element)
            
        else:
            
            dict_to_return = {}
            
            for key in keys:
                
                dict_to_return[key] = decode_components_input_element(source_component, element[key])
                
            return dict_to_return
                
    
    elif isinstance(element, list):
                
        return [decode_components_input_element(source_component, value) for value in element]
            
    else:
        return element
    

def decode_components_exposed_values(component : Component, source_component : Component, component_dict : dict):
    
    if "exposed_values" in component_dict: #if there are exposed values
    
        saved_exposed_values = component_dict["exposed_values"]

        for exposed_values_key, exposed_value in saved_exposed_values.items():
            component.values[exposed_values_key] = decode_components_input_element(source_component, exposed_value)

    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_exposed_values(child_component, source_component, component_dict["child_components"][i])
    


def decode_components_input(component : Component, source_component : Component, component_dict : dict):
    
    '''
        From a component to change
        The source_component, which is the root of the component tree
        And a dictionary which specifies the component and its children
    '''
    
    
    input_to_pass = {}
    
    component_dict_input = component_dict["input"]
    
    for key in component_dict_input.keys():
        input_to_pass[key] = decode_components_input_element(source_component, component_dict_input[key])
        
    component.pass_input(input_to_pass)
    
    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_input(child_component, source_component, component_dict["child_components"][i])
                

def decode_components_from_dict(dict : dict):
    
    '''Decodes only the components and child components from a dictionary representation of a system.'''
    
    component_type_name = dict["__type__"]
    component_name = dict["name"]
        
    component_type = get_class_from(component_type_name)
    
    component : Component = component_type()
    component.name = component_name
    
    try:
    
        child_components = dict.get("child_components") 

        if child_components != None:

            for child_dict in child_components:

                decoded_child_component = decode_components_from_dict(child_dict)

                component.define_component_as_child(decoded_child_component)
    
    except Exception as e:
        raise e
            
    return component
    
# EXPOSED DECODING METHODS ---------------------------------------------------------------------------------   

def set_values_of_dict_in_component(component : Component, dict_representation : dict):
    
    '''Receives a component with children and uses the dictionary to set its values (input, exposed values, ...)'''
    
    source_component = component.get_source_component()

    decode_components_input(component, source_component, dict_representation)
    decode_components_exposed_values(component, source_component, dict_representation)
    
    
    
    
def gen_component_from_dict(dict_representation) -> Component:
    '''Returns a component, decoding it from a dictionary representation of a system'''
    
    source_component_with_children = decode_components_from_dict(dict_representation)
    
    set_values_of_dict_in_component(source_component_with_children, dict_representation)
    
    return source_component_with_children



def dict_from_json_string(json_string) -> dict:
    '''Returns a dictionary representation of a system, decoded from a json string'''
    return json.loads(json_string)



def component_from_json_string(json_string) -> Component:
    '''Returns a component, reading it from a json_string'''
    
    dict_representation = dict_from_json_string(json_string)
        
    return gen_component_from_dict(dict_representation)


def component_from_tuple_definition(tuple_definition) -> Component:

    '''
    Generates a component from a tuple definition (Component_definition, input)
    
    '''
     
    class_definition = tuple_definition[0]
    
    if len(tuple_definition) > 2:
        raise Exception(f"Tuple definition has more than 2 elements, but it should have only 2, got {len(tuple_definition)}")
    
    elif len(tuple_definition) == 2:
        input = tuple_definition[1]
        
        if not isinstance(input, dict):
            raise Exception(f"Input in tuple is not a dict, but {type(input)}")
    
    else:
        input = {} #input was not passed
        
    class_of_component : type = get_class_from(class_definition)
        
    component_to_return = class_of_component(input=input)
    return component_to_return
        
        

def gen_component_from(definition :  Union[Component, dict, str, tuple], parent_component_for_generated : Component = None) -> Component:
    
    '''Generates a component from a definition or returns it if it is already a component'''

    if isinstance(definition, Component):
        return definition
    
    else: # gen component if it was a definition
    
        if isinstance(definition, dict):
            generated_component = gen_component_from_dict(definition)

        elif isinstance(definition, str):

            if os.path.exists(definition):
                return gen_component_from_path(definition)

            else: # is json
                generated_component =  component_from_json_string(definition)

        elif isinstance(definition, tuple) or isinstance(definition, list):

            generated_component =  component_from_tuple_definition(definition)

        else:
            raise Exception(f"Definition is not a Component, dict, str or tuple | list, but {type(definition)}")
        
        if parent_component_for_generated is not None:
            parent_component_for_generated.define_component_as_child(generated_component)
    
        return generated_component
    
    

def gen_component_from_path(path):
    
    if os.path.isdir(path):
        return gen_component_in_directory(path)
    
    elif os.path.isfile(path):
        return gen_component_in_file_path(path)
    
    else:
        raise ValueError(f"Path '{path}' is neither a file nor a directory.")
    
    

def gen_component_in_directory(dir_path):
    
    configuration_file = os.path.join(dir_path, CONFIGURATION_FILE_NAME)

    if os.path.exists(configuration_file):
        return gen_component_in_file_path(configuration_file)
    
    component_loaded_file = os.path.join(dir_path, LOADED_COMPONENT_FILE_NAME)
    
    if os.path.exists(component_loaded_file):
        return gen_component_in_file_path(component_loaded_file)

    raise Exception("No component defined in folder")

def gen_component_in_file_path(file_path):
    
    if file_path.endswith('.json'):
        
        with open(file_path, 'r') as f:
            str_to_gen_from = f.read()
            return component_from_json_string(str_to_gen_from)

    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


    raise Exception("Not supported file to generate component from")

# OTHER METHODS ---------------------------------------------------------------------------------

def get_child_dict_from_index_localization(component_dict, localization : int) -> dict:
    
    if "child_components" in component_dict:
        
        try:
        
            child_components : list = component_dict["child_components"]
        
        except IndexError:
            
            raise IndexError(f"Localization index {localization} out of range for component with children: {child_components}")
            
            
        
        return child_components[localization]        

    return None

def get_child_dict_from_str_localization(component_dict, localization : str) -> dict:
    
    if "child_components" in component_dict:
        
        child_components : list = component_dict["child_components"]
        
        for child_component in child_components:
            
            if child_component["name"] == localization:
                return child_component

        return None     

    return None


def get_child_dict_from_localization(component_dict, localization) -> dict:
    
    if isinstance(localization, int):
        return get_child_dict_from_index_localization(component_dict, localization)
    
    elif isinstance(localization, str):
        return get_child_dict_from_str_localization(component_dict, localization)
    
    elif isinstance(localization, list):
        
        if len(localization) == 0:
            return component_dict
                    
        elif len(localization) == 1:
            return get_child_dict_from_localization(component_dict, localization[0])
        
        else:
        
            child_component_dict = get_child_dict_from_localization(component_dict, localization[0])

            return get_child_dict_from_localization(child_component_dict, localization[1:])
    
    else:
        raise Exception(f"Localization is not an int or a str, but {type(localization)}")
    
    
    


# COMPONENT CONFIGURATION UTILS ------------------------------------


def save_configuration(component : Component, config_directory, config_filename=CONFIGURATION_FILE_NAME, save_exposed_values=False, ignore_defaults=True):
        
        json_str = json_string_of_component(component, save_exposed_values=save_exposed_values, ignore_defaults=ignore_defaults)
        
        path_to_save_configuration = os.path.join(config_directory, config_filename)

        write_text_to_file(filename=path_to_save_configuration, text=json_str)  