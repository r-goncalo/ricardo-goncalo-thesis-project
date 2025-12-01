
from typing import Union
from automl.component import Component

from automl.utils.json_utils.json_component_utils import get_child_dict_from_localization, is_valid_component_tuple_definition
from automl.utils.json_utils.custom_json_logic import CustomJsonLogic, register_custom_strategy
import optuna

from automl.core.localizations import get_component_by_localization_list, get_last_collection_where_value_is, safe_get
from automl.loggers.global_logger import globalWriteLine
from automl.core.input_management import InputSignature
from automl.utils.class_util import get_class_from
from automl.hp_opt.hyperparameter_suggestion import HyperparameterSuggestion


def generate_hyperparameter_suggestion_list_for_config_dict(config_dict):

    return _gen_hp_suggestions_for_collection(config_dict, current_localization=[])



def _gen_hp_suggestions_for_collection(collection, current_localization):

    if isinstance(collection, dict):
        return _gen_hp_suggestions_for_config(collection, current_localization)
    
    elif isinstance(collection, list):
        return _gen_hp_suggestions_for_list(collection, current_localization)
    
    else:
        raise Exception(f"Invalid type to generate hyperparameter suggestions")


def _gen_hp_suggestions_for_list(config_list : list, current_localization):

    to_return = []

    for i in range(len(config_list)):

        value = config_list[i]

        new_localization = [*current_localization, i]

        to_return = gen_hp_suggestions_for_new_localization_and_value(to_return, new_localization, value)

    return to_return


def _gen_hp_suggestions_for_config(config_dict : dict, current_localization):

    to_return = []

    for key, value in config_dict.items():

        new_localization = [*current_localization, key]

        to_return = gen_hp_suggestions_for_new_localization_and_value(to_return, new_localization, value)


    return to_return

def gen_hp_suggestions_for_new_localization_and_value(suggestions, new_localization, value):

        if isinstance(value, Component):
            suggestions = [ *suggestions,
                *gen_hp_suggestions_for_instanced_component(value, new_localization)]
            
        elif is_valid_component_tuple_definition(value):
            suggestions = [ *suggestions,
                *gen_hp_suggestions_for_tuple_definition(value, new_localization)]

        elif isinstance(value, (dict, list)):
            suggestions = [*suggestions, *_gen_hp_suggestions_for_collection(value, new_localization)]

        return suggestions

def gen_hp_suggestions_for_instanced_component(component : Component, current_localization):

    class_definition = type(component)
    input_of_component = component.input

    return [
            *gen_hp_suggestions_for_component_and_input(
                                class_definition, 
                                input_of_component,
                                current_localization
                            ),
            
             *_gen_hp_suggestions_for_collection(input_of_component, [*current_localization, "__get_input_value__"])
                            
            ]


def gen_hp_suggestions_for_tuple_definition(tuple_definition, current_localization):

    (class_definition, input_of_component) = tuple_definition
    class_definition : type[Component] = get_class_from(class_definition)

    return [
            *gen_hp_suggestions_for_component_and_input(
                                class_definition, 
                                input_of_component,
                                current_localization
                            ),
            
             *_gen_hp_suggestions_for_collection(input_of_component, [*current_localization, 1])
                            
            ]



def gen_hp_suggestions_for_component_and_input(component_class : type[Component], input, current_localization):

    to_return = []

    for key, value in component_class.get_schema_parameters_signatures().items():

        if value.custom_dict != None:
        
            hyperparameter_suggestion_in_parameter_signature = value.custom_dict.get("hyperparameter_suggestion", None)
    
            if hyperparameter_suggestion_in_parameter_signature is not None:
                to_return.append(
                    gen_hp_suggestion_for_parameter_schema(hyperparameter_suggestion_in_parameter_signature, [*current_localization, "__get_input_value__", key])
                )

    return to_return



def gen_hp_suggestion_for_parameter_schema(hyperparameter_suggestion, current_localization):

    if isinstance(hyperparameter_suggestion, HyperparameterSuggestion):
        return hyperparameter_suggestion.clone()
    
    else:
        hyperparameter_name = current_localization[len(current_localization) - 1]
        hyperparameter_localizations = [current_localization]
        value_suggestion = hyperparameter_suggestion

        return HyperparameterSuggestion(hyperparameter_name, hyperparameter_localizations, value_suggestion)

