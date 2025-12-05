

from automl.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
import optuna
from automl.utils.json_utils.custom_json_logic import register_custom_strategy

class SingleHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''
        An Hyperparameter Suggestion that is a simple optuna wrapper
    '''
    
    def __init__(self, name : str = '', value_suggestion=None, hyperparameter_localizations=None):
        
        if value_suggestion is None:
            raise Exception("Value suggestion must be different than none")

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        self.value_suggestion = value_suggestion

    # SUGGESTIONS ------------------------------------    
        
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
        
    # JSON ENCODING DECODING ------------------------------------------------------------------------

    def clone(self):
        return SingleHyperparameterSuggestion(
            name=self.base_name,
            hyperparameter_localizations=self.hyperparameter_localizations,
            value_suggestion=self.value_suggestion
        )        

    def to_dict(self) -> dict:
        
        return {
                    **super().to_dict(),
                    "suggestion" : self.value_suggestion
            }
                    
            
    def from_dict(dict : dict, decode_elements_fun, source_component): # we have no use for the function for nested components, there are none
        return SingleHyperparameterSuggestion(name=dict["name"], hyperparameter_localizations=dict.get("localizations", None), value_suggestion=dict["suggestion"])



register_custom_strategy(SingleHyperparameterSuggestion, SingleHyperparameterSuggestion)
