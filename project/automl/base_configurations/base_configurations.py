import os

from automl.utils.json_component_utils import component_from_dict
from automl.component import Schema

from typing import Literal

import automl.base_configurations.basic_rl as basic_rl

def load_configuration(configuration_name : Literal["basic_rl"]) -> Schema:
    
    '''Loads a pre-made configuration'''
    
    if configuration_name == "basic_rl":
    
        return component_from_dict(basic_rl.config_dict())
    