import importlib

def get_class_from(class_definition):
    
    if isinstance(class_definition, type):
        return class_definition
    
    elif isinstance(class_definition, str):
        return get_class_from_string(class_definition)
    
    else:
        raise Exception(f"Clas definition {class_definition} is neither of type str nor type")

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
        raise ValueError(f"{class_string} has invalid class string format")
    
    full_path = class_string[len("<class '"):-len("'>")]
    
    # Split into module path and class name
    *module_parts, class_name = full_path.split('.')
    module_name = '.'.join(module_parts)
    
    # Dynamically import the module and get the class
    try:
        
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Unable to locate module '{module_name}' in class '{class_string}': {e}")
    
    except Exception as e:
        raise Exception(f"Problem when getting class with string {class_string}: {e}")
        

def __get_all_subclasses_recursive(cls):

    direct_subs = cls.__subclasses__()
    return direct_subs + [g for s in direct_subs for g in get_all_subclasses(s)]
    

def get_all_subclasses(cls):
    
    '''Returns a list of all subclasses of a class, including indirect ones.'''
    
    return [cls, *__get_all_subclasses_recursive(cls)]
    
