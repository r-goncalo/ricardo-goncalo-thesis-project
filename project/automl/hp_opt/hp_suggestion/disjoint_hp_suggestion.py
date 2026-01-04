

from automl.component import Component
import optuna
from automl.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automl.utils.json_utils.custom_json_logic import register_custom_strategy


class DisjointHyperparameterSuggestion(HyperparameterSuggestion):
    
    '''A class which defines a range of values a specific hyperparameter group can have'''
    
    def __init__(self, name : str = '',  disjoint_hyperparameter_suggestions : list[HyperparameterSuggestion]=None, hyperparameter_localizations=None):
        
        if disjoint_hyperparameter_suggestions is None:
            raise Exception(f"Disjoint hyperparameter suggestions must not be none")
        
        self.disjoint_hyperparameter_suggestions : dict[str, HyperparameterSuggestion] = {}
        for hyperparameter_suggestion in disjoint_hyperparameter_suggestions:
            self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion.name] = hyperparameter_suggestion.clone()

        super().__init__(name=name, hyperparameter_localizations=hyperparameter_localizations)

        if self.hyperparameter_localizations != None:
            for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
                
                if hyperparameter_suggestion.hyperparameter_localizations != None:
                    raise Exception(f"Both disjoint hyperparameter suggestion and child have localizations defined")
                
                hyperparameter_suggestion.change_localizations(self.hyperparameter_localizations)


    
    def setup_names(self):
        super().setup_names()
        
        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            suggestion_base_name = hyperparameter_suggestion.get_base_name()

            hyperparameter_suggestion.change_name(
                f"{self.name}_{suggestion_base_name}"
            )

        
    
    def _set_suggested_value_in_component(self, suggested_value, component : Component, hyperparameter_localizer):
        
        '''Sets the suggested value in the component, using the localization'''

        (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use) = suggested_value
    
        self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]._set_suggested_value_in_component(suggested_value_of_suggestion_to_use, component, hyperparameter_localizer)



    
    def _set_suggested_value_in_dict(self, suggested_value, component_dict : dict, hyperparameter_localizer):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use) = suggested_value
    
        self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]._set_suggested_value_in_dict(suggested_value_of_suggestion_to_use, component_dict, hyperparameter_localizer)
    

        
    def _make_suggestion(self, trial : optuna.Trial):
        
        '''Creates a suggested value for an hyperparameter group and changes the corresponding objects, children of the source_component'''
        
        hyperparameter_suggestion_to_use_name = trial.suggest_categorical(self.name, self.disjoint_hyperparameter_suggestions.keys())

        hyperparameter_suggestion_to_use = self.disjoint_hyperparameter_suggestions[hyperparameter_suggestion_to_use_name]

        suggested_value_of_suggestion_to_use = hyperparameter_suggestion_to_use.make_suggestion(trial)           
                
            
        return (hyperparameter_suggestion_to_use_name, suggested_value_of_suggestion_to_use)


    def _try_get_already_suggested_value_in_component(self, component : Component, hyperparameter_localizer):
        
        '''Gets the suggested value in the component, using the localization'''

        suggested_value = None

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            try:
                suggested_value = hyperparameter_suggestion._try_get_already_suggested_value_in_component(component, hyperparameter_localizer)
            except:
                suggested_value = None

            if suggested_value != None:
                break
        
        return suggested_value


    
    def _try_get_suggested_value_in_dict(self, component_dict : dict, hyperparameter_localizer):
    
        '''Sets the suggested value in the dictionary representing a component, using the localization'''

        suggested_value = None

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            
            try:
                suggested_value = hyperparameter_suggestion._try_get_suggested_value_in_dict(component_dict, hyperparameter_localizer)
            except:
                suggested_value = None

            if suggested_value != None:
                break
        
        return suggested_value
    
    # JSON ENCODING DECODING ------------------------------------------------------------------------
            

    def clone(self):

        return DisjointHyperparameterSuggestion(
            name = self.base_name,
            disjoint_hyperparameter_suggestions = [disjoint_hyperparameter_suggestion.clone()  for disjoint_hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()]
        )
            
    def to_dict(self) -> dict:
        
        
        return  {

            **super().to_dict(),
            "suggestions" : [
                hyperparameter_suggestion for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values()
                ]
            
        }
                            
            
    def from_dict(dict : dict, element_type, decode_elements_fun, source_component):

        '''Decodes object dict'''
                
        suggestion_dicts =  [ hyperparameter_suggestion_dict for hyperparameter_suggestion_dict in dict["suggestions"] ]

        disjoint_to_return = DisjointHyperparameterSuggestion(
            name=dict["name"], 
            hyperparameter_localizations=dict.get("localizations", None),
            disjoint_hyperparameter_suggestions=decode_elements_fun(source_component, suggestion_dicts)) # it is done this way to be able to deal with nested components


        return disjoint_to_return
    
    def get_nested_hyperparameter_suggestions(self) -> list[HyperparameterSuggestion]:

        to_return = super().get_nested_hyperparameter_suggestions()

        for hyperparameter_suggestion in self.disjoint_hyperparameter_suggestions.values():
            to_return = [*to_return, * hyperparameter_suggestion.get_nested_hyperparameter_suggestions()]

        return to_return
        
register_custom_strategy(DisjointHyperparameterSuggestion, DisjointHyperparameterSuggestion)