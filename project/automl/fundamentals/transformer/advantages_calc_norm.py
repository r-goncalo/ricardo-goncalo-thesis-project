

from automl.component import Component
from automl.core.advanced_input_management import ComponentListInputSignature
from automl.core.input_management import InputSignature
from automl.fundamentals.transformer.transformer import Transformer


class AdvantagesCalcNorm(Transformer):

    '''Used to calculate and normalize advantages on whole data, before using mini_batch'''

    parameters_signature = {

    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()


    def transform_data(self, data):
        '''Method called to transform data'''
    
