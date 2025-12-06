
from typing import Union
from automl.component import Component

from automl.utils.json_utils.json_component_utils import decode_components_input_element, get_child_dict_from_localization
from automl.utils.json_utils.custom_json_logic import CustomJsonLogic, register_custom_strategy
import optuna

from automl.core.localizations import get_component_by_localization_list, get_last_collection_where_value_is, safe_get
from automl.loggers.global_logger import globalWriteLine

class HyperparameterSuggestion(CustomJsonLogic):
    
    '''
    A class which defines a range of values a specific hyperparameter group can have

    It can have defined localizations, in which case it is able to set suggested values in those localizations or get already suggested values
    '''
    
    def __init__(self, name : str = '', hyperparameter_localizations=None):
        
        self.name = name
        self.base_name = name
        self.hyperparameter_localizations = hyperparameter_localizations

        self.setup_names()

    # NAMES -----------------------------------------


    def get_name(self):
        '''Gets the current name of the Hyperparameter Suggestion, should be unique'''
        return self.name

    def get_base_name(self):
        return self.base_name

    def change_name(self, new_name):
        '''Changes the name and sets up any changes accross children'''
        self.name = new_name        
        self.setup_names()

    def setup_names(self):
        '''Sets up the names of child hyperparameter suggestions'''
        pass

    # SUGGESTIONS ------------------------------------

    def make_suggestion(self, trial : optuna.Trial):
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''

    # LOCALIZATIONS ---------------------------------------

    def get_localizations(self):
        return self.hyperparameter_localizations
    
    def change_localizations(self, new_localizations):
        self.hyperparameter_localizations = new_localizations

    # SET VALUE IN LOCALIZATION -----------------------------------

    def set_suggested_value(self, suggested_value, component_definition : Union[Component, dict], localization=None):
        
        '''Sets the suggested value in the component (or component input), using the localization'''
        
        localization = self.hyperparameter_localizations if localization is None else localization

        if localization is None:
            raise Exception(f"No localization specified in function call nor in object to set suggested value in component")

        
        if isinstance(component_definition, Component):
            self._set_suggested_value_in_component(suggested_value, component_definition, localization)
            
        elif isinstance(component_definition, dict):
            self._set_suggested_value_in_dict(suggested_value, component_definition, localization)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}") 
    
    
    def _set_suggested_value_in_component(self, suggested_value, component : Component, hyperparameter_localizations):
        
        '''Sets the suggested value in the component, using the localization'''
    
        for hyperparameter_localizer in hyperparameter_localizations:
            
            # gets the component change 
            component_to_change : Component = get_component_by_localization_list(component, hyperparameter_localizer[:-1]) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{component_localizer}> in {component.name}")   
            
            component_to_change.pass_input({hyperparameter_localizer[-1] : suggested_value})



    def _set_suggested_value_in_dict(self, suggested_value, component_dict : dict, hyperparameter_localizations):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        for hyperparameter_localizer in hyperparameter_localizations:
            
            component_dict : dict = get_last_collection_where_value_is(component_dict, hyperparameter_localizer)

            component_dict[hyperparameter_localizer[-1]] = suggested_value
                


    # GET VALUE IN LOCALIZATION ---------------------------------------------------------

    def try_get_suggested_optuna_values(self, component_definition, localizaiton=None):

        localization = self.hyperparameter_localizations if localization is None else localization

        if localization is None:
            raise Exception(f"No localization specified in function call nor in object to get suggested value in component")

        return self._try_get_suggested_optuna_values(component_definition, localization)
    


    def _try_get_suggested_optuna_values(self, component_definition, localization):
        
        return {self.name : self.try_get_suggested_value(component_definition, localization)}



    def try_get_suggested_value(self, component_definition : Union[Component, dict], localization=None):
        
        '''Gets the suggested value in the component (or component input), using the localization'''
        
        localization = self.hyperparameter_localizations if localization is None else localization

        if localization is None:
            raise Exception(f"No localization specified in function call nor in object to get suggested value in component")

        return self._try_get_suggested_value(component_definition, localization)


    def _try_get_suggested_value(self, component_definition : Union[Component, dict], localization):
        
        '''Gets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            return self._try_get_already_suggested_value_in_component(component_definition, localization)
            
        elif isinstance(component_definition, dict):
            return self._try_get_suggested_value_in_dict(component_definition, localization)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}") 


    def _try_get_already_suggested_value_in_component(self, component : Component, hyperparameter_localizations):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for hyperparameter_localizer in hyperparameter_localizations:
            
            component_to_change : Component = get_component_by_localization_list(component, hyperparameter_localizer[:-1]) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{hyperparameter_localizer}> in {component.name}")   
            
            localization_suggested_value = component_to_change.get_input_value(hyperparameter_localizer[-1])

            if suggested_value != None and localization_suggested_value != suggested_value:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and localizations have different values. This is still not implemented, and the value will be treated as if it is non existent")
                return None

            if localization_suggested_value == None:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and one of localizations does not have any value. This is still not implemented, and the value will be treated as if it is non existent")
                return None
            
            if suggested_value == None:
                suggested_value = localization_suggested_value

            #if we reach here, localization suggested value exists and is equal to previous value

        return suggested_value
    


    
    def _try_get_suggested_value_in_dict(self, component_dict : dict, hyperparameter_localizations):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for hyperparameter_localizer in hyperparameter_localizations:
            
            colleciton_where_hyperparameter_is : dict = get_last_collection_where_value_is(component_dict, hyperparameter_localizer)

            print(f"Trying to get value from {hyperparameter_localizer}")

            localization_suggested_value = safe_get(colleciton_where_hyperparameter_is, hyperparameter_localizer[-1], None)

            if suggested_value != None and localization_suggested_value != suggested_value:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and localizations have different values. This is still not implemented, and the value will be treated as if it is non existent")
                return None

            if localization_suggested_value == None:
                globalWriteLine(f"WARNING: Trying to get already suggested value in configuration with name {self.name}, and one of localizations does not have any value. This is still not implemented, and the value will be treated as if it is non existent")
                return None
            
            if suggested_value == None:
                suggested_value = localization_suggested_value

            #if we reach here, localization suggested value exists and is equal to previous value

        return suggested_value


    
    def _try_get_already_passed_input_to_component_input(self, component_input, hyperparameter_localizer):
        
        '''Passes the suggested value to the component input, using the localization'''
        
        current_input_dict = get_last_collection_where_value_is(component_input, hyperparameter_localizer)

        try:
            return safe_get(current_input_dict, hyperparameter_localizer[len(hyperparameter_localizer) - 1], default_value=None)
        
        except Exception as e:
            raise Exception(f"Exception when trying to get last indice ({hyperparameter_localizer[len(hyperparameter_localizer) - 1]}) of hyperparameter_localizer: {hyperparameter_localizer}, {e}")
        
        
    # JSON ENCODING DECODING ------------------------------------------------------------------------

            
    def clone(self):
        return HyperparameterSuggestion(name=self.name, hyperparameter_localizations=self.hyperparameter_localizations)

    def to_dict(self) -> dict:

        to_return = {"name" : self.base_name,
                     "__type__" : type(self)}

        if self.hyperparameter_localizations != None:
            to_return["localizations"] = self.hyperparameter_localizations
        
        return to_return
                    
            
    def from_dict(dict : dict, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
        return HyperparameterSuggestion(dict["name"], dict.get("localizations", None))
    

    def get_nested_hyperparameter_suggestions(self) -> list:
        '''Gets the hyperparameter suggestions that are nested in this one'''
        return [self]


register_custom_strategy(HyperparameterSuggestion, HyperparameterSuggestion)





        