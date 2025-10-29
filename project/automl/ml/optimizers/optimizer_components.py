from automl.component import Component, InputSignature, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature, LookableInputSignature
from automl.ml.models.model_components import ModelComponent
from automl.basic_components.dynamic_value import get_value_or_dynamic_value
from automl.core.advanced_input_utils import get_value_of_type_or_component
from automl.loggers.logger_component import ComponentWithLogging
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


from abc import abstractmethod

class OptimizerSchema(Component):
    
    parameters_signature = {"model" : ComponentInputSignature(ignore_at_serialization=True)}
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()
        
        self.model : ModelComponent = ComponentInputSignature.get_value_from_input(self, "model")


        
        
    @abstractmethod
    def optimize_model(self, predicted, correct) -> None:
        
        '''
            Optimizes the model based on the prediction(s) and correct value(s)
            
            Args:
                predicted is the prediced value(s) of the model for a given input
                correct is the correct value(s) for the given input
                
        '''
        
        pass

    def optimize_with_loss(self, loss):
        pass

    def clear_optimizer_gradients(self):
        pass

    def optimize_with_backward_pass_done(self):
        pass


class AdamOptimizer(OptimizerSchema, ComponentWithLogging):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                       "learning_rate" : InputSignature(default_value=0.001),
                       "amsgrad" : InputSignature(default_value=False),

                       "clip_grad_value" : InputSignature(mandatory=False, description="If defined, it clips the gradients to the given value"),
                       "clip_grad_norm" : InputSignature(mandatory=False, description="If defined, it clips the gradients to the given norm"),

                       "linear_decay_learning_rate_with_final_input_value_of" : LookableInputSignature(mandatory=False)
                       
                       
                       }    
    
    exposed_values = {
        "optimizations_done" : 0
    }
    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.lg.writeLine(f"This exists with input:\n{self.input}")
        
        self.params = self.model.get_model_params() #gets the model parameters to optimize
                
        self.torch_adam_opt = optim.Adam(params=self.params,lr=self.input["learning_rate"], amsgrad=self.input["amsgrad"])

        self._initialize_decays()
        self._initialize_grad_clip()

    # INITIALIZATION --------------------------------------------------------------------------

    def _initialize_decays(self):

        self.lr_scheduler = None

        self.linear_decay_learning_rate_with_final_input_value_of = LookableInputSignature.get_value_from_input(self, "linear_decay_learning_rate_with_final_input_value_of", (int)) 

        if self.linear_decay_learning_rate_with_final_input_value_of != None:

            self.lg.writeLine(f"LR will have linear decay, using a linear decay till 0 and a predicted final number of optimizations of {self.linear_decay_learning_rate_with_final_input_value_of}")

            self.lr_scheduler = LambdaLR(self.torch_adam_opt, lr_lambda=lambda step: 1 - step / self.linear_decay_learning_rate_with_final_input_value_of)

    def _initialize_grad_clip(self):

        self.clip_grad_value = get_value_of_type_or_component(self, "clip_grad_value", float)

        if self.clip_grad_value != None:
            self.lg.writeLine(f"Optimizer has clip grad value of type: {type(self.clip_grad_value)}, with value {self.clip_grad_value}")
        
        self.clip_grad_norm = get_value_of_type_or_component(self, "clip_grad_norm", float)

        if self.clip_grad_norm != None:
            self.lg.writeLine(f"Optimizer has clip grad norm of type: {type(self.clip_grad_norm)}, with value {self.clip_grad_value}")

    # EXPOSED METHODS --------------------------------------------------------------------------
    
    @requires_input_proccess
    def optimize_model(self, predicted, correct) -> None:
                    
        # Compute Huber loss TODO : The loss should not be hard calculated like this
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted, correct)
        
        self.optimize_with_loss(loss)


    @requires_input_proccess
    def optimize_with_loss(self, loss):

        self.clear_optimizer_gradients()

        #Optimize the model
        loss.backward()

        self.optimize_with_backward_pass_done()


    @requires_input_proccess
    def clear_optimizer_gradients(self):
        self.torch_adam_opt.zero_grad() #clears previous accumulated gradients in the optimizer


    
    def optimize_with_backward_pass_done(self):

        # In-place gradient clipping
        if self.clip_grad_value != None:
            nn.utils.clip_grad_value_(self.params, get_value_or_dynamic_value(self.clip_grad_value))
        
        if self.clip_grad_norm != None:
            nn.utils.clip_grad_norm_(self.params, get_value_or_dynamic_value(self.clip_grad_norm))
        
        self.torch_adam_opt.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.values["optimizations_done"])

        self.values["optimizations_done"] = self.values["optimizations_done"] + 1
        

class SimpleSGDOptimizer(OptimizerSchema):

    parameters_signature = {
        "learning_rate": InputSignature(default_value=0.01)
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        # Get model parameters
        self.params = self.model.get_model_params()

        # Define SGD optimizer
        self.sgd_optimizer = optim.SGD(self.params, lr=self.input["learning_rate"])

    @requires_input_proccess
    def optimize_model(self, predicted, correct) -> None:
        super().optimize_model(predicted, correct)

        # Use simple Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(predicted, correct)

        # Zero previous gradients
        self.sgd_optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        self.sgd_optimizer.step()