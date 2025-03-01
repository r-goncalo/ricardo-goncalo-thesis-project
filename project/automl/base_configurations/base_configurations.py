import os

from automl.utils.json_component_utils import component_from_dict
from automl.component import Schema

from typing import Literal

import automl.base_configurations.basic_rl as basic_rl

def load_configuration(configuration_name : Literal["basic_rl"]) -> Schema:
    
    '''Loads a pre-made configuration'''
    
    return component_from_dict(load_configuration_dict(configuration_name))
    

def load_configuration_dict(configuration_name : Literal["basic_rl"]) -> dict:
    
    if configuration_name == "basic_rl":
    
        return basic_rl.config_dict()
    