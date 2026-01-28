import os
from automl.basic_components.state_management import StatefulComponent
from automl.loggers.global_logger import globalWriteLine
from automl.loggers.logger_component import ComponentWithLogging
import torch
import torch.nn as nn

from automl.component import Component, InputSignature, requires_input_proccess
from automl.ml.models.model_components import ModelComponent

from automl.ml.models.torch_model_components import TorchModelComponent

class TorchModelComponentDebug(TorchModelComponent):

    is_debug_schema = True

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
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