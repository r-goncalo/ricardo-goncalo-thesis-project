import json
from typing import Union

from automl.component import Component, InputSignature, InputMetaData


from automl.utils.class_util import get_class_from_string

from enum import Enum

# ENCODING --------------------------------------------------

class ComponentInputElementsEncoder(json.JSONEncoder):
    
    '''Encodes elements of a component input, which can be a component (defined by its localization) or a primitive type'''
    
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
                    
                    serialized_value = json.dumps(input[key], cls=ComponentInputElementsEncoder, source_component=self.source_component) #for each value in input, loads
                    
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
                
                serialized_value = json.dumps(exposed_values[key], cls=ComponentInputElementsEncoder, source_component=self.source_component) #for each value in exposed_value, loads
                   
                if serialized_value != 'null':
                    toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
            

class ComponentEncoder(json.JSONEncoder):
    
    
    '''Encodes the definition of a component, focusing on its child components'''
    
    def __init__(self, *args, ignore_defaults : bool, save_exposed_values : bool, source_component, **kwargs):
        
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


def json_string_of_component(component, ignore_defaults = False, save_exposed_values = False):
    return json.dumps(component, cls=ComponentEncoder, indent=4, ignore_defaults=ignore_defaults, save_exposed_values = save_exposed_values)

# DECODING --------------------------------------------------------------------------
    

def decode_components_input_element(source_component : Component, element):
    
    '''
    Decodes an element in a dictionary of a component
    One of the things it does is get a component with a localization and set it
    '''
        
    if isinstance(element, dict):
        keys = element.keys()
        
        if "__type__" in keys:
            
            class_of_component : type = get_class_from_string(element['__type__'])
            
            if issubclass(class_of_component, Component): #if it is a Schema
                
                if "localization" in keys:
                        
                    component_to_return = source_component.get_child_by_localization(element["localization"])
                    
                elif "name" in keys:
                    
                    component_to_return = source_component.get_child_by_name(element["name"])
                        
            return component_to_return
        
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
    
    
    saved_exposed_values = component_dict["exposed_values"]
    
    for exposed_values_key, exposed_value in saved_exposed_values.items():
        component.values[exposed_values_key] = decode_components_input_element(source_component, exposed_value)

    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_input(child_component, source_component, component_dict["child_components"][i])
    


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
        
    component_type = get_class_from_string(component_type_name)
    
    component : Component = component_type()
    component.name = component_name
    
    try:
    
        child_components = dict.get("child_components") 

        if child_components != None:

            for child_dict in child_components:

                decoded_child_component = decode_components_from_dict(child_dict)

                component.define_component_as_child(decoded_child_component)
    
    except Exception as e:
        print(f"Exception on decoding child components of component named {component_name} with type {component_type_name}")
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
        input = {}
        
    if isinstance(class_definition, str): #if if is a class
        class_of_component : type = get_class_from_string(class_definition)
        return class_of_component(input=input)

        
    elif isinstance(class_definition, type): # if it is a class
        class_of_component : type = class_definition
        return class_of_component(input=input)
    
    else: #if the first element of the tuple is another type of definition
        component_to_return = gen_component_from(class_definition)
        component_to_return.pass_input(input=input)
        return component_to_return
        
        

def gen_component_from(definition :  Union[Component, dict, str, tuple]) -> Component:
    
    '''Generates a component from a definition'''

    if isinstance(definition, Component):
        return definition
    
    elif isinstance(definition, dict):
        return gen_component_from_dict(definition)
    
    elif isinstance(definition, str):
        return component_from_json_string(definition)
    
    elif isinstance(definition, tuple):
        
        return component_from_tuple_definition(definition)
    
    else:
        raise Exception(f"Definition is not a Component, dict, str or tuple, but {type(definition)}")
        
    
# OTHER METHODS ---------------------------------------------------------------------------------

def get_child_dict_from_index_localization(component_dict, localization : int) -> dict:
    
    if "child_components" in component_dict:
        
        child_components : list = component_dict["child_components"]
        
        return child_components[localization]        

    return None

def get_child_dict_from_str_localization(component_dict, localization : int) -> dict:
    
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
        
        if len(localization) == 1:
            return get_child_dict_from_localization(component_dict, localization[0])
        
        child_component_dict = get_child_dict_from_localization(component_dict, localization[0])
        
        return get_child_dict_from_localization(child_component_dict, localization[1:])
    
    else:
        raise Exception(f"Localization is not an int or a str, but {type(localization)}")