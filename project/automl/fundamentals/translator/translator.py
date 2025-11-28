

from automl.component import Component
from automl.core.advanced_input_management import ComponentListInputSignature


class Translator(Component):

    '''A strategy that translates data with a preditermined type and shape to another'''


    def translate_state(self, state):
        return state
    
    def get_shape(self, original_shape):
        return original_shape
    


class TranslatorSequence(Translator):

    '''A strategy that translates data with a preditermined type and shape to another'''

    parameters_signature = {
        "translators_sequence" : ComponentListInputSignature()
    }

    def proccess_input(self):

        super().proccess_input()

        self.translators_sequence : list[Translator] = self.get_input_value("translators_sequence")

        for translator in self.translators_sequence:
            translator.proccess_input_if_not_proccesd()


    def translate_state(self, state):
        
        current_state = state

        for translator in self.translators_sequence:
            current_state = translator.translate_state(current_state)

        return current_state
    

    def get_shape(self, original_shape):

        current_shape = original_shape

        for translator in self.translators_sequence:
            current_shape = translator.get_shape(current_shape)

        return current_shape