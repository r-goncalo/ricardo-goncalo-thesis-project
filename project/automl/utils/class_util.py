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
        raise ImportError(f"Unable to locate class '{class_string}': {e}")
    
    except Exception as e:
        raise Exception(f"Problem when getting class with string {class_string}: {e}")
        