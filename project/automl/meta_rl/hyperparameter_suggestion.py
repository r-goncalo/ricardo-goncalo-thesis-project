
from automl.component import Schema

import optuna



class HyperparameterSuggestion():
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str, hyperparameter_localizations, value_suggestion):
        
        self.name = name
        self.hyperparameter_localizations = hyperparameter_localizations
        self.value_suggestion = value_suggestion
        
    def make_suggestion(self, source_component : Schema, trial : optuna.Trial):
        
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
        
        
        for (component_localizer, hyperparameter_localizer) in self.hyperparameter_localizations:
            
            component_to_change : Schema = source_component.get_child_component(component_localizer)    
            
            self.pass_input(component_to_change, hyperparameter_localizer, suggested_value)
            
                
            
        return suggested_value
    
    def pass_input(self, component_to_change : Schema, hyperparameter_localizer, suggested_value):
        
        print("Passing input to component: " + component_to_change.name)
                
        if isinstance(hyperparameter_localizer, str):
            component_to_change.pass_input({hyperparameter_localizer : suggested_value})
            return        
        
        elif len(hyperparameter_localizer) == 1:
            component_to_change.pass_input({hyperparameter_localizer[0] : suggested_value})
            return
        
        current_input_dict = component_to_change.input

        
        for i in range(0, len(hyperparameter_localizer) - 1):
            
            try:
                current_input_dict = current_input_dict[hyperparameter_localizer[i]]
        
            except KeyError as e:
                raise KeyError(f'Error when locating hyperparameter using localization {hyperparameter_localizer}, in key {hyperparameter_localizer[i]}')
            
        current_input_dict[len(hyperparameter_localizer) - 1] = suggested_value