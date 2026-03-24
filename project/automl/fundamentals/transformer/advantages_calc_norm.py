

from automl.component import Component
from automl.core.advanced_input_management import ComponentListParameterSignature
from automl.core.input_management import ParameterSignature
from automl.fundamentals.transformer.transformer import Transformer


class AdvantagesCalcNorm(Transformer):

    '''Used to calculate and normalize advantages on whole data, before using mini_batch'''

    parameters_signature = {

    }

    def _process_input_internal(self):
        super()._process_input_internal()


    def transform_data(self, data):
        '''Method called to transform data'''
    
