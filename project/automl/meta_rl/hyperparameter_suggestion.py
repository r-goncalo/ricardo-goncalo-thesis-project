
from typing import Union
from automl.component import Component

from automl.utils.json_component_utils import get_child_dict_from_localization
import optuna



class HyperparameterSuggestion():
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str, hyperparameter_localizations, value_suggestion):
        
        self.name = name
        self.hyperparameter_localizations = hyperparameter_localizations
        self.value_suggestion = value_suggestion
        
    
    def set_suggested_value_in_component(self, suggested_value, component : Component):
        
        '''Sets the suggested value in the component, using the localization'''
    
        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_to_change : Component = component.get_child_component(component_localizer) 
            
            if component_to_change == None:
                raise Exception(f"Could not find component with localization <{component_localizer}> in {component.name}")   
            
            self.pass_input(component_to_change, hyperparameter_localizer, suggested_value)



    
    def set_suggested_value_in_dict(self, suggested_value, component_dict : dict):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_dict : dict = get_child_dict_from_localization(component_dict, component_localizer)
            
            if not "input" in component_dict:
                component_input_dict = {}
            
            else:
                component_input_dict = component_dict["input"]

            if component_dict == None:
                raise Exception(f"Could not find component with localization <{component_localizer}>")   

            self.pass_input_to_component_input(component_input_dict, hyperparameter_localizer, suggested_value)
        
    
    
    
    def set_suggested_value(self, suggested_value, component_definition : Union[Component, dict]):
        
        '''Sets the suggested value in the component (or component input), using the localization'''
        
        if isinstance(component_definition, Component):
            self.set_suggested_value_in_component(suggested_value, component_definition)
            
        elif isinstance(component_definition, dict):
            self.set_suggested_value_in_dict(suggested_value, component_definition)
            
        else:   
            raise Exception(f"Component definition is not a Component or a dict, but {type(component_definition)}") 
        
        
        
        
        
        
    def make_suggestion(self, source_component : Union[Component, dict], trial : optuna.Trial):
        
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
        
        
        self.set_suggested_value(suggested_value, source_component)
            
                
            
        return suggested_value
    
    def pass_input_to_component_input(self, component_input, hyperparameter_localizer, suggested_value):
        
        '''Passes the suggested value to the component input, using the localization'''
        
        current_input_dict = component_input
        
        for i in range(0, len(hyperparameter_localizer) - 1):
        
            try:
                current_input_dict = current_input_dict[hyperparameter_localizer[i]]
    
            except KeyError as e:
                raise KeyError(f'Error when locating hyperparameter using localization {hyperparameter_localizer}, in key {hyperparameter_localizer[i]}, for current component input {current_input_dict}')
        
        current_input_dict[hyperparameter_localizer[len(hyperparameter_localizer) - 1]] = suggested_value
        
        
    
    def pass_input(self, component_to_change : Component, hyperparameter_localizer, suggested_value):
                        
        '''Passes the suggested value to the component, using the localization'''
                        
        if isinstance(hyperparameter_localizer, str):
            component_to_change.pass_input({hyperparameter_localizer : suggested_value})
            return        
        
        elif len(hyperparameter_localizer) == 1:
            component_to_change.pass_input({hyperparameter_localizer[0] : suggested_value})
            return
        
        else:
            
            self.pass_input_to_component_input(component_to_change.input, hyperparameter_localizer, suggested_value)
            
            
            
    def to_dict(self):
        
        dict_to_return = {
            "name" : self.name,
            "localizations" : self.hyperparameter_localizations,
            "suggestion" : self.value_suggestion
            
            }
        
        return dict_to_return
            
            
    def from_dict(dict):
                
        return HyperparameterSuggestion(dict["name"], dict["localizations"], dict["suggestion"])
        
        
        