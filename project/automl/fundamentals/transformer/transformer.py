

from automl.component import Component
from automl.core.advanced_input_management import ComponentListInputSignature
from automl.core.input_management import InputSignature


class Transformer(Component):

    '''A strategy that transforms data, such as normalizing it'''

    parameters_signature = {

    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()


    def transform_data(self, data : dict):
        '''Method called to transform data'''
    
