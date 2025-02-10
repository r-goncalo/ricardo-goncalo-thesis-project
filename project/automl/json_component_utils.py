import json

from .component import Schema, InputSignature




# ENCODING --------------------------------------------------


class ComponentInputElementsEncoder(json.JSONEncoder):
    
    def default(self, obj):
                        
        if isinstance(obj, Schema):

            return {
                "__type__": type(obj).__name__,
                "name" : obj.name,
                "localization" : obj.get_localization()
            }
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
        
        try:    
            return super().default(obj)
        
        except:
            
            return None
        

class ComponentInputEncoder(json.JSONEncoder):
    
    '''Used with component to encode its input (not the input itself)'''
    
    def default(self, obj):
        
        if isinstance(obj, Schema):
            
            input = obj.input
            
            toReturn = {}
            
            for key in input.keys():
                                  
                parameters_signature : InputSignature = obj.get_parameter_signature(key)
                                
                if not parameters_signature.ignore_at_serialization:
                    
                    serialized_value = json.dumps(input[key], cls=ComponentInputElementsEncoder) #for each value in input, loads
                    
                    if serialized_value != 'null':
                        toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
            

class ComponentEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        if isinstance(obj, Schema):
            
            toReturn = {
                "__type__": str(type(obj)),
                "name": obj.name,
                "input": json.loads(json.dumps(obj, cls= ComponentInputEncoder))
                }
            
            if len(obj.child_components) > 0:
                
                toReturn["child_components"] = self.default(obj.child_components)
                        
            return toReturn
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
            
        return None


def json_string_of_component(component):
    return json.dumps(component, cls=ComponentEncoder, indent=4)

# DECODING --------------------------------------------------------------------------

def get_component_from_source(source_component, localization):
    
    current_component : Schema = source_component
    
    for index in localization:
        current_component = current_component.child_components[index]
        
    return current_component
    

def decode_components_input_element(source_component : Schema, element):
    
    if isinstance(element, dict):
        keys = element.keys()
        
        if "__type__" in keys and "localization" in keys:
            
            return get_component_from_source(source_component, element["localization"])
        
        else:
            
            dict_to_return = {}
            
            for key in keys:
                
                dict_to_return[key] = decode_components_input_element(source_component, element[key])
                
            return dict_to_return
                
    
    elif isinstance(element, list):
        
        return [decode_components_input_element(source_component, value) for value in element]
    
    else:
        return element
    


def decode_components_input(component : Schema, source_component : Schema, component_dict : dict):
    
    input_to_pass = {}
    
    component_dict_input = component_dict["input"]
    
    for key in component_dict_input.keys():
        input_to_pass[key] = decode_components_input_element(source_component, component_dict_input[key])
        
    component.pass_input(input_to_pass)
    
    for i in range(0, len(component.child_components)):
        
        child_component = component.child_components[i]
            
        decode_components_input(child_component, source_component, component_dict["child_components"][i])
        
        

def decode_components_from_dict(dict : dict):
    
    component_type_name = dict["__type__"]
    component_name = dict["name"]
        
    component_type = get_class_from_string(component_type_name)
    
    component : Schema = component_type()
    component.name = component_name
    
    child_components = dict.get("child_components") 
    
    if child_components != None:
        
        for child_dict in child_components:
            component.child_components.append(decode_components_from_dict(child_dict)) 
            
    return component
    
    
def component_from_json_string(json_string):
    
    dict_representation = json.loads(json_string)
    
    source_component = decode_components_from_dict(dict_representation)
    
    decode_components_input(source_component, source_component, dict_representation)
    
    return source_component
    
    
# UTIL -----------------------------------------------------------

import importlib

def get_class_from_string(class_string: str):
    """
    Returns the class type from its string representation.

    Args:
        class_string (str): A string like "<class 'module.submodule.ClassName'>".

    Returns:
        type: The class type if found, otherwise raises an error.
    """
    # Extract the full path of the class
    if not class_string.startswith("<class '") or not class_string.endswith("'>"):
        raise ValueError("Invalid class string format.")
    
    full_path = class_string[len("<class '"):-len("'>")]
    
    # Split into module path and class name
    *module_parts, class_name = full_path.split('.')
    module_name = '.'.join(module_parts)
    
    # Dynamically import the module and get the class
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Unable to locate class '{class_string}': {e}")