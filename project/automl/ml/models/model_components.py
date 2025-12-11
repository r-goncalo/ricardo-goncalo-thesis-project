from automl.component import Component, InputSignature, requires_input_proccess


class ModelComponent(Component):
        
    parameters_signature = {
        "input_shape": InputSignature(mandatory=False, description="Used for models which can still change their input shape"),
        "output_shape": InputSignature(mandatory=False, description="Used for models which can still change their output shape"), #output shape can have a default value
    }    
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self._setup_values()
        

    def _setup_values(self):

        '''Sets up basic values from the input, such as input shapes and such'''

        self.input_shape = self.get_input_value("input_shape")
        self.output_shape = self.get_input_value("output_shape")
    
    
    def predict(self, state):
        pass
    
    @requires_input_proccess
    def get_model_params(self):
        '''returns a list of model parameters'''
        pass
    
    @requires_input_proccess
    def random_prediction(self):
        pass
    
    
    @requires_input_proccess            
    def update_model_with_target(self, target_model, target_model_weight):
        pass
    
    @requires_input_proccess
    def get_model_input_shape(self):
        return self.input_shape
    
    @requires_input_proccess
    def get_model_output_shape(self):
        return self.output_shape
        
    
    
    
    