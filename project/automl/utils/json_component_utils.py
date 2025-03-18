import json

from automl.component import Schema, InputSignature, InputMetaData


from automl.utils.class_util import get_class_from_string

# ENCODING --------------------------------------------------


class ComponentInputElementsEncoder(json.JSONEncoder):
    
    def default(self, obj):
                        
        if isinstance(obj, Schema):

            return {
                "__type__": str(type(obj)),
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
    
    def __init__(self, *args, ignore_defaults : bool, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
    
    def default(self, obj):
        
        if isinstance(obj, Schema):
            
            input = obj.input
            
            toReturn = {}
            
            for key in input.keys():
                
                parameter_meta : InputMetaData = obj.get_input_meta()[key]
                parameters_signature : InputSignature = parameter_meta.parameter_signature
                                
                if ( not parameters_signature.ignore_at_serialization ) and (not ( ( not parameter_meta.was_custom_value_passed() ) and self.ignore_defaults )):
                    
                    serialized_value = json.dumps(input[key], cls=ComponentInputElementsEncoder) #for each value in input, loads
                    
                    if serialized_value != 'null':
                        toReturn[key]  = json.loads(serialized_value)
                    
            return toReturn
                
        
        else:
            raise Exception('Was expecting an object of type Component')
            

class ComponentEncoder(json.JSONEncoder):
    
    def __init__(self, *args, ignore_defaults : bool, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.ignore_defaults = ignore_defaults
    
    def default(self, obj):
        
        if isinstance(obj, Schema):
            
            toReturn = {
                "__type__": str(type(obj)),
                "name": obj.name,
                "input": json.loads(json.dumps(obj, cls= ComponentInputEncoder, ignore_defaults=self.ignore_defaults))
                }
            
            if len(obj.child_components) > 0:
                
                toReturn["child_components"] = self.default(obj.child_components)
                        
            return toReturn
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
            
        return None


def json_string_of_component(component, ignore_defaults = False):
    return json.dumps(component, cls=ComponentEncoder, indent=4, ignore_defaults=ignore_defaults)

# DECODING --------------------------------------------------------------------------
    

def decode_components_input_element(source_component : Schema, element):
        
    if isinstance(element, dict):
        keys = element.keys()
        
        if "__type__" in keys:
            
            class_of_component : type = get_class_from_string(element['__type__'])
            
            if issubclass(class_of_component, Schema): #if it is a Schema
                
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
    
def component_from_dict(dict):
    '''Returns a component, decoding it from a dictionary representation of a system'''
    
    source_component = decode_components_from_dict(dict)
    
    decode_components_input(source_component, source_component, dict)
    
    return source_component



def dict_from_json_string(json_string):
    '''Returns a dictionary representation of a system, decoded from a json string'''
    return json.loads(json_string)



def component_from_json_string(json_string):
    '''Returns a component, reading it from a json_string'''
    
    dict_representation = dict_from_json_string(json_string)
    
    return component_from_dict(dict_representation)
     

    
    
