import os

from automl.utils.json_component_utils import component_from_dict
from automl.component import Component

from typing import Literal

import automl.base_configurations.basic_dqn as basic_dqn
import automl.base_configurations.basic_ppo as basic_ppo


def load_configuration(configuration_name : Literal["basic_dqn"], *args, **kwargs) -> Component:
    
    '''Loads a pre-made configuration'''
    
    return component_from_dict(load_configuration_dict(configuration_name, *args, **kwargs))
    

def load_configuration_dict(configuration_name : Literal["basic_dqn", "basic_ppo"], *args, **kwargs) -> dict:
    
    if configuration_name == "basic_dqn":
    
        return basic_dqn.config_dict(*args, **kwargs)
    
    elif configuration_name == "basic_ppo":
        
        return basic_ppo.config_dict(*args, **kwargs)

    