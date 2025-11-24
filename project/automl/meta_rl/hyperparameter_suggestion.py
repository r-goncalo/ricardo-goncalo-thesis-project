
from typing import Union
from automl.component import Component

from automl.utils.json_utils.json_component_utils import get_child_dict_from_localization
from automl.utils.json_utils.custom_json_logic import CustomJsonLogic, register_custom_strategy
import optuna

from automl.core.localizations import get_component_by_localization_list, get_last_collection_where_value_is, safe_get
from automl.loggers.global_logger import globalWriteLine

class HyperparameterSuggestion(CustomJsonLogic):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str, hyperparameter_localizations, value_suggestion):
        
        self.name = name
        self.hyperparameter_localizations = hyperparameter_localizations
        self.value_suggestion = value_suggestion
        
    
    def _set_suggested_value_in_component(self, suggested_value, component : Component):
        
        '''Sets the suggested value in the component, using the localization'''
    
        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            # gets the component change 
            component_to_change : Component = get_component_by_localization_list(component_localizer, hyperparameter_localizer) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{component_localizer}> in {component.name}")   
            
            self._pass_input(component_to_change, hyperparameter_localizer, suggested_value)



    
    def _set_suggested_value_in_dict(self, suggested_value, component_dict : dict):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_dict : dict = get_child_dict_from_localization(component_dict, component_localizer)
            
            if not "input" in component_dict:
                component_input_dict = {}
            
            else:
                component_input_dict = component_dict["input"]

            if component_dict == None:
                raise Exception(f"Could not find component with localization <{component_localizer}>")   

            self._pass_input_to_component_input(component_input_dict, hyperparameter_localizer, suggested_value)
        
    
    
    
    def set_suggested_value(self, suggested_value, component_definition : Union[Component, dict]):
        
        '''Sets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            self._set_suggested_value_in_component(suggested_value, component_definition)
            
        elif isinstance(component_definition, dict):
            self._set_suggested_value_in_dict(suggested_value, component_definition)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}") 
        
        
        
    def make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        (type_of_suggestion, kwargs) = self.value_suggestion
        
        if type_of_suggestion == 'float': 
            suggested_value = trial.suggest_float(self.name, **kwargs)
        
        elif type_of_suggestion == 'int':
            suggested_value = trial.suggest_int(self.name, **kwargs)
        
        elif type_of_suggestion == 'cat':
            suggested_value = trial.suggest_categorical(self.name, **kwargs)
        
        else:
            raise Exception(f"Invalid value suggestion with name {self.name}")            
                
            
        return suggested_value
    
    


    def _pass_input_to_component_input(self, component_input, hyperparameter_localizer, suggested_value):
        
        '''Passes the suggested value to the component input, using the localization'''
        
        current_input_dict = get_last_collection_where_value_is(component_input, hyperparameter_localizer)

        try:

            current_input_dict[hyperparameter_localizer[len(hyperparameter_localizer) - 1]] = suggested_value

        except Exception as e:

            raise Exception(f"Exception when setting last indice ({hyperparameter_localizer[len(hyperparameter_localizer) - 1]}) of hyperparameter_localizer: {hyperparameter_localizer}, {e}")

        
        
    
    def _pass_input(self, component_to_change : Component, hyperparameter_localizer, suggested_value):
                        
        '''Passes the suggested value to the component, using the localization'''
                        
        if isinstance(hyperparameter_localizer, str):
            component_to_change.pass_input({hyperparameter_localizer : suggested_value})
            return        
        
        elif len(hyperparameter_localizer) == 1:
            component_to_change.pass_input({hyperparameter_localizer[0] : suggested_value})
            return
        
        else:
            
            self._pass_input_to_component_input(component_to_change.input, hyperparameter_localizer, suggested_value)


    def _try_get_already_suggested_value_in_component(self, component : Component):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_to_change : Component = get_component_by_localization_list(component, component_localizer) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{component_localizer}> in {component.name}")   
            
            localization_suggested_value = self._try_get_already_passed_input(component_to_change, hyperparameter_localizer)

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


    
    def _try_get_suggested_value_in_dict(self, component_dict : dict):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_dict : dict = get_child_dict_from_localization(component_dict, component_localizer)
            
            if not "input" in component_dict:
                component_input_dict = {}
            
            else:
                component_input_dict = component_dict["input"]

            if component_dict == None:
                raise Exception(f"Could not find component with localization <{component_localizer}>")   


            localization_suggested_value = self._try_get_already_passed_input_to_component_input(component_input_dict, hyperparameter_localizer)
        

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

    
    def try_get_suggested_value(self, component_definition : Union[Component, dict]):
        
        '''Sets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            return self._try_get_already_suggested_value_in_component(component_definition)
            
        elif isinstance(component_definition, dict):
            return self._try_get_suggested_value_in_dict(component_definition)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}") 

    
    def _try_get_already_passed_input_to_component_input(self, component_input, hyperparameter_localizer):
        
        '''Passes the suggested value to the component input, using the localization'''
        
        current_input_dict = get_last_collection_where_value_is(component_input, hyperparameter_localizer)

        try:
            return safe_get(current_input_dict, hyperparameter_localizer[len(hyperparameter_localizer) - 1], default_value=None)
        
        except Exception as e:
            raise Exception(f"Exception when trying to get last indice ({hyperparameter_localizer[len(hyperparameter_localizer) - 1]}) of hyperparameter_localizer: {hyperparameter_localizer}, {e}")
        
    
    def _try_get_already_passed_input(self, component_to_change : Component, hyperparameter_localizer):
                        
        '''Passes the suggested value to the component, using the localization'''
                        
        if isinstance(hyperparameter_localizer, str):

            return component_to_change.input.get(hyperparameter_localizer, None)     
        
        elif len(hyperparameter_localizer) == 1:
            return component_to_change.input.get(hyperparameter_localizer[0], None)   
        
        else:
            
            return self._try_get_already_passed_input_to_component_input(component_to_change.input, hyperparameter_localizer)
            
        
    # JSON ENCODING DECODING ------------------------------------------------------------------------

            
    def clone(self):
        return HyperparameterSuggestion(
            name=self.name,
            hyperparameter_localizations=self.hyperparameter_localizations,
            value_suggestion=self.value_suggestion
        )        

    def to_dict(self):
        
        return {

                    "name" : self.name,
                    "localizations" : self.hyperparameter_localizations,
                    "suggestion" : self.value_suggestion

            }
                    
            
    def from_dict(dict, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
                
        return HyperparameterSuggestion(dict["name"], dict["localizations"], dict["suggestion"])
    
register_custom_strategy(HyperparameterSuggestion, HyperparameterSuggestion)


class DisjointHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str, disjoint_hyperparameter_suggestions : list[HyperparameterSuggestion]):
        
        self.name = name

        self.disjoint_hyperparameter_suggestions : dict[str, HyperparameterSuggestion] = {}
        for hyperparameter_suggestion in disjoint_hyperparameter_suggestions:
            self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion.name] = hyperparameter_suggestion
        
    
    def _set_suggested_value_in_component(self, suggested_value, component : Component):
        
        '''Sets the suggested value in the component, using the localization'''

        (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use) = suggested_value
    
        self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]._set_suggested_value_in_component(suggested_value_of_suggestion_to_use, component)



    
    def _set_suggested_value_in_dict(self, suggested_value, component_dict : dict):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use) = suggested_value
    
        self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]._set_suggested_value_in_dict(suggested_value_of_suggestion_to_use, component_dict)
    

        
    def make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        hyperparameter_suggestion_to_use_name = trial.suggest_categorical(self.name, self.disjoint_hyperparameter_suggestions.keys())

        hyperparameter_suggestion_to_use = self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]

        suggested_value_of_suggestion_to_use = hyperparameter_suggestion_to_use.make_suggestion(trial)           
                
            
        return (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use)


    def _try_get_already_suggested_value_in_component(self, component : Component):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            try:
                suggested_value = hyperparameter_suggestion._try_get_already_suggested_value_in_component(component)
            except:
                suggested_value = None

            if suggested_value != None:
                break
        
        return suggested_value


    
    def _try_get_suggested_value_in_dict(self, component_dict : dict):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            try:
                suggested_value = hyperparameter_suggestion._try_get_suggested_value_in_dict(component_dict)
            except:
                suggested_value = None

            if suggested_value != None:
                break
        
        return suggested_value
    
    # JSON ENCODING DECODING ------------------------------------------------------------------------
            

    def clone(self):

        return DisjointHyperparameterSuggestion(
            name = self.name,
            disjoint_hyperparameter_suggestions = [disjoint_hyperparameter_suggestion.clone()  for disjoint_hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()]
        )
            
    def to_dict(self):
        
        
        return  {

            "name" : self.name,
            "suggestions" : [
                hyperparameter_suggestion for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()
                ]
            
        }
                            
            
    def from_dict(dict, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        suggestion_dicts =  [ hyperparameter_suggestion_dict for hyperparameter_suggestion_dict in dict["suggestions"] ]

        disjoint_to_return = DisjointHyperparameterSuggestion(
            name=dict["name"], 
            disjoint_hyperparameter_suggestions=decode_elements_fun(source_component, suggestion_dicts)) # it is done this way to be able to deal with nested components


        return disjoint_to_return
        
register_custom_strategy(DisjointHyperparameterSuggestion, DisjointHyperparameterSuggestion)



class VariableListHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str, name_of_list : str, kwargs_for_list, hyperparameter_suggestion_for_list : HyperparameterSuggestion):
        
        self.name = name

        self.name_len_list = name_of_list
        self.kwargs_for_list = kwargs_for_list

        self.hyperparameter_suggestion_for_list : HyperparameterSuggestion =  hyperparameter_suggestion_for_list


    def clone(self):
        
        return VariableListHyperparameterSuggestion(
            name=self.name,
            name_of_list=self.name_len_list,
            kwargs_for_list = self.kwargs_for_list,
            hyperparameter_suggestion_for_list=self.hyperparameter_suggestion_for_list.clone()
        )


    def make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        list_len= trial.suggest_int(self.name_len_list, **self.kwargs_for_list)

        list_to_return = []

        for i in range(list_len):

            suggestion_to_put_in_list = self.hyperparameter_suggestion_for_list.clone()

            suggestion_to_put_in_list.name = f"{suggestion_to_put_in_list.name}_{i}"

            list_to_return.append(suggestion_to_put_in_list)
                
            
        return list_to_return


    
    # JSON ENCODING DECODING ------------------------------------------------------------------------
            
            
    def to_dict(self):
        
        
        return  {
            "name" : self.name,
            "name_of_list" : self.name_len_list,
            "kwargs_for_list" : self.kwargs_for_list,
            "hyperparameter_suggestion_for_list" : self.hyperparameter_suggestion_for_list
        }
                            
            
    def from_dict(dict, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        return VariableListHyperparameterSuggestion(
            name=dict["name"],
            name_of_list=dict["name_of_list"], 
            kwargs_for_list=dict["kwargs_for_list"],
            hyperparameter_suggestion_for_list=decode_elements_fun(source_component, dict["hyperparameter_suggestion_for_list"])

        )

        
register_custom_strategy(VariableListHyperparameterSuggestion, VariableListHyperparameterSuggestion)
        