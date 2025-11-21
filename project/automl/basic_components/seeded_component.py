


from automl.component import Component
from automl.core.input_management import InputSignature
from automl.basic_components.state_management import StatefulComponent
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.random_utils import SEED_GLOBAL_LOGGER, do_full_setup_of_seed, generate_seed
from automl.loggers.global_logger import globalWriteLine


class SeededComponent(Component):
    
    parameters_signature = {
                       "seed" : InputSignature(generator=lambda self : generate_seed()),
                       "do_full_setup_of_seed" : InputSignature(ignore_at_serialization=True, default_value=False, description="If it is supposed to setup the seed for things like torch, python, and so on")
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()
        
        self._seed = self.get_input_value("seed")
        self._do_full_setup_of_seed = self.get_input_value("do_full_setup_of_seed")

        globalWriteLine(f"{self}: Seed is {self._seed}", file=SEED_GLOBAL_LOGGER)
                
        if self._do_full_setup_of_seed:
            globalWriteLine(f"{self}: Activating full setup of seed", file=SEED_GLOBAL_LOGGER)
            do_full_setup_of_seed(self._seed)