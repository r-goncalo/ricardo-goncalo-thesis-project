


from automl.component import Component
from automl.core.input_management import InputSignature
from automl.basic_components.state_management import StatefulComponent
from automl.loggers.logger_component import ComponentWithLogging
from automl.utils.random_utils import SEED_GLOBAL_LOGGER, do_full_setup_of_single_seed, generate_seed, setup_seed_from_dict_configuration
from automl.loggers.global_logger import globalWriteLine


class SeededComponent(Component):

    '''
    A component which uses a seed in its computations
    '''
    
    parameters_signature = {
                       "seed" : InputSignature(generator=lambda self : generate_seed()),
                       "do_full_setup_of_seed" : InputSignature(mandatory=False, description="If it is supposed to setup the seed for things like torch, python, and so on")
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def generate_and_setup_input_seed(self, seed = None, to_do_full_setup_of_seed= None):
        '''Meant to be called before initialization by a parent component that may want to define the seed of this'''
        self.pass_input({"seed": generate_seed() if seed is None else seed })

        if to_do_full_setup_of_seed is not None:
            self.pass_input({"do_full_setup_of_seed" : to_do_full_setup_of_seed})


    def _proccess_input_internal(self): #this is the best method to have initialization done right after
        
        super()._proccess_input_internal()
        
        
        self_do_full_setup_of_seed = self.get_input_value("do_full_setup_of_seed")
        self.seed = self.get_input_value("seed")
        globalWriteLine(f"{self.name}: Seed is {self.seed}", file=SEED_GLOBAL_LOGGER)
                
        if self_do_full_setup_of_seed != None and self_do_full_setup_of_seed != False:

            globalWriteLine(f"{self.name}: Activating full setup of seed", file=SEED_GLOBAL_LOGGER)
            
            if self_do_full_setup_of_seed == True:
                seed_config = do_full_setup_of_single_seed(self.seed)
                

            elif isinstance(self_do_full_setup_of_seed, dict):
                seed_config = setup_seed_from_dict_configuration(self_do_full_setup_of_seed)

            else:
                raise Exception(f"Non valid type for key 'do_full_setup_of_seed': {type(seed_config)}")
            
            self.input["do_full_setup_of_seed"] = False if seed_config is None else seed_config

        


        #globalWriteLine(f"{self.name}: Random state is:\n\n{get_random_state()}\n\n", file=SEED_GLOBAL_LOGGER)


    