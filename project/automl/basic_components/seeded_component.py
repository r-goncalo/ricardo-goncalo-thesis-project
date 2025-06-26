


from automl.core.input_management import InputSignature
from automl.basic_components.state_management import StatefulComponent
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.random_utils import do_full_setup_of_seed, generate_seed


class SeededComponent(ComponentWithLogging, StatefulComponent):
    
    parameters_signature = {
                       "seed" : InputSignature(generator=lambda self : generate_seed()),
                       "do_full_setup_of_seed" : InputSignature(ignore_at_serialization=True, default_value=False, description="If it is supposed to setup the seed for things like torch, python, and so on")
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super().proccess_input_internal()
                
        if self.input["do_full_setup_of_seed"]:
            do_full_setup_of_seed(self.input["seed"])
        