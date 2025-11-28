

from automl.component import Component


class Translator(Component):

    '''A strategy that translates data with a preditermined type and shape to another'''


    def translate_state(self, state):
        return state
    
    def get_shape(self, original_shape):
        return original_shape