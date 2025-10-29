



from automl.component import Component
from automl.core.input_management import InputSignature
from automl.core.advanced_input_management import ComponentInputSignature



def get_value_of_type_or_component(component_with_input : Component, key : str, desired_type : type) -> Component:

    '''Gets either a value in the key with the desired type or, if that fails, tries to get a component. If it does not exist, returns none'''

    value_in_input = InputSignature.get_value_from_input(component_with_input, key)

    if value_in_input is None:
        return None

    if isinstance(value_in_input, desired_type):
        return value_in_input

    try:
        component_to_return = ComponentInputSignature.get_value_from_input(component_with_input, key)
        return component_to_return
    
    except Exception as e:

        print(f"WARNING: Value with key {key} in component {component_with_input.name} is {value_in_input}, which could not be get as a component nor is it of the desired type: {desired_type}")

    
    return value_in_input