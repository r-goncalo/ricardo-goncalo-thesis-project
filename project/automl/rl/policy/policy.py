from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from

from automl.utils.shapes_util import single_action_shape
from automl.loggers.logger_component import ComponentWithLogging


class PolicyInterface(Component):
    
    '''
    A policy interface, defining only the methods necessary for a policy
    This abstracts wrappers and our own implemented strategies
    '''
        
    def get_policy_output_shape(self):
        raise NotImplementedError()
    
    def get_policy_input_shape(self):
        raise NotImplementedError()
        
    def predict(self, state):
        raise NotImplementedError()
    
    def random_prediction(self, state):
        raise NotImplementedError()



class Policy(PolicyInterface, ComponentWithLogging):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''
        
    parameters_signature = {
        
        "model" : ComponentInputSignature(mandatory=False),
        
        "state_shape": InputSignature(),
        "action_shape": InputSignature(),
        "device" : InputSignature(get_from_parent=True, ignore_at_serialization=True)
    }   

    exposed_values = {"model" : 0}
    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.input_state_shape = self.get_input_value("state_shape")
        self.output_action_shape = self.get_input_value("action_shape")

        self.policy_output_shape = single_action_shape(self.output_action_shape)

        self.lg.writeLine(f"Action shape {self.output_action_shape} means policy chooses action value with shape {self.policy_output_shape}")

        self.device = self.get_input_value("device")

        self._initialize_model()

        self._setup_model()
        
        
        
        
        
                

        
    def _setup_model(self):
        self.model.pass_input({
            "input_shape" : self.input_state_shape, 
            "output_shape" : self.output_action_shape, 
            "device" : self.device
            }) 

    
    @requires_input_proccess
    def get_policy_output_shape(self):
        return self.policy_output_shape
        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        pass

    def _initialize_model(self):

        self.lg.writeLine(f"Initializing policy model...")
        
        if hasattr(self, "model"):

            self.lg.writeLine(f"Model already in attributes, using that one...")

        elif self.values["model"] != 0:

            self.lg.writeLine(f"Model for policy already in values, using that one...")
            self.model : ModelComponent = self.values["model"]

        else:

            self.lg.writeLine(f"No model in attributes nor defined in values, using model in input...")

            self.model = self.get_input_value("model")

            if self.model == None:
                self.lg.writeLine(f"No model found in attributes")
                raise Exception(f"Policy had no model in input nor saved in its values")
            

        self.lg.writeLine(f"Ended policy model setup")
        self.values["model"] = self.model
        
    
    def random_prediction(self, state):
        pass