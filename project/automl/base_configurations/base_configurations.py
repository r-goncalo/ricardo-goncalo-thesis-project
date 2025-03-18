import os

from automl.utils.json_component_utils import component_from_dict
from automl.component import Schema

from typing import Literal

import automl.base_configurations.basic_rl as basic_rl

def load_configuration(configuration_name : Literal["basic_rl"], *args, **kwargs) -> Schema:
    
    '''Loads a pre-made configuration'''
    
    return component_from_dict(load_configuration_dict(configuration_name, *args, **kwargs))
    

def load_configuration_dict(configuration_name : Literal["basic_rl"], *args, **kwargs) -> dict:
    
    if configuration_name == "basic_rl":
    
        return basic_rl.config_dict(*args, **kwargs)
    