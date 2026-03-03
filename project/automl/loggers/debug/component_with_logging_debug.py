from automl.loggers.logger_component import ComponentWithLogging


class ComponentWithLoggingDebug(ComponentWithLogging):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    is_debug_schema = True
    
    parameters_signature = {
                       }


    def _proccess_input_internal(self): #this is the best method to have initialization done right after
            
        super()._proccess_input_internal()

    def pass_input(self, input):
        super().pass_input(input)

        if hasattr(self, "_lg"):
            self.lg.writeLine(f"Received input: {input}")
        

    def _try_look_input_in_attribute(self, input_key, attribute_name):

        if not hasattr(self, "_lg"):
            return super()._try_look_input_in_attribute(input_key, attribute_name)

        self.lg.writeLine(f"Trying to look for attribute '{attribute_name}' for input '{input_key}'")

        to_return = super()._try_look_input_in_attribute(input_key, attribute_name)

        if to_return == None:
            self.lg.writeLine(f"Did not have attribute '{attribute_name}' for input '{input_key}'")

        else:
            self.lg.writeLine(f"Attribute '{attribute_name}' for input '{input_key}' was found")

        return to_return
    

    def _try_look_input_in_values(self, input_key, value_name):

        if not hasattr(self, "_lg"):
            return super()._try_look_input_in_values(input_key, value_name)

        self.lg.writeLine(f"Trying to look for value '{value_name}' for input '{input_key}'")

        to_return = super()._try_look_input_in_values(input_key, value_name)

        if to_return == None:
            self.lg.writeLine(f"Did not have value '{value_name}' for input '{input_key}'")

        else:
            self.lg.writeLine(f"Value '{value_name}' for input '{input_key}' was found") 

        return to_return
    

    def on_parent_component_defined(self):
        super().on_parent_component_defined()

        if hasattr(self, "_lg"):
            self.lg.writeLine(f"New parent component was defined: {self.parent_component}")


    def _generate_artifact_directory(self):

        super()._generate_artifact_directory()

    def clone(self, *args, **kwargs):

        self.lg.writeLine(f"Cloning component...")

        return super().clone(*args, **kwargs)
