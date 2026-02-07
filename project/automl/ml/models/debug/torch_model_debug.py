import os
from automl.basic_components.state_management import StatefulComponent
from automl.loggers.global_logger import globalWriteLine
from automl.loggers.logger_component import ComponentWithLogging
import torch
import torch.nn as nn

from automl.component import Component, InputSignature, requires_input_proccess
from automl.ml.models.model_components import ModelComponent

from automl.ml.models.torch_model_components import TorchModelComponent

from automl.ml.models.torch_model_utils import model_parameter_distance_by_params, split_shared_params

class TorchModelComponentDebug(TorchModelComponent):

    is_debug_schema = True

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
        "note_model_difference_on_init" : InputSignature(default_value=True)
    }    

    exposed_values = {
    }
    
    def _proccess_input_internal(self):

        
        super()._proccess_input_internal()

        

    @requires_input_proccess
    def predict(self, state):
        self.lg.writeLine(f"Predicting value for state with shape {state.shape}...", file="model_predictions.txt")
        to_return = super().predict(state)
        self.lg.writeLine(f"Predicted value with shape {to_return.shape}: {to_return}\n", file="model_predictions.txt")
        return to_return
    
    def _execute_model_initialization_strategy(self):

        self.__note_model_difference_on_init = False if self.model_initialization_strategy is None else self.get_input_value("note_model_difference_on_init")

        if self.__note_model_difference_on_init:
            olds_params = torch.cat([p.flatten() for p in self.model.parameters()])

        super()._execute_model_initialization_strategy()

        if self.__note_model_difference_on_init:

            new_params = torch.cat([p.flatten() for p in self.model.parameters()])

            l2_distance, avg_distance, cosine_sim = model_parameter_distance_by_params(olds_params, new_params)

            self.lg.writeLine(f"Difference between old and new model params, ater executing init strategy: l2: {l2_distance}, avg: {avg_distance}, cos: {cosine_sim}")

        
    
    def clone(self, save_in_parent=True, input_for_clone=None, is_deep_clone=True):

        cloned_component = super().clone(save_in_parent=save_in_parent, input_for_clone=input_for_clone, is_deep_clone=is_deep_clone)

        shared_params, self_only, cloned_only = split_shared_params(self, cloned_component)

        self.lg.writeLine(f"Cloned component: Shared params: {len(shared_params)}, Self only: {len(self_only)}, Cloned only: {len(cloned_only)}")

        return cloned_component