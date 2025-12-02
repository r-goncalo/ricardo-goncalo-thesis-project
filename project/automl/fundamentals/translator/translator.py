

from automl.component import Component
from automl.core.advanced_input_management import ComponentListInputSignature
from automl.core.input_management import InputSignature


class Translator(Component):

    '''A strategy that translates data with a preditermined type and shape to another'''

    parameters_signature = {

        "original_shape" : InputSignature(mandatory=False),
        "in_place_translation" : InputSignature(default_value=False),
        "buffered_operations" : InputSignature(default_value=False),
        
        
        "device" : InputSignature(default_value="", ignore_at_serialization=True, get_from_parent=True)

    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.in_place_translation = self.get_input_value("in_place_translation")

        self.buffered_operations = self.get_input_value("buffered_operations")

        self.device = self.get_input_value("device")

        self.original_shape = self.get_input_value("original_shape")
        self._setup_shape_cache()


    def _setup_shape_cache(self):

        if self.original_shape is not None:
            self.new_shape = self.get_shape(self.original_shape)

        else:
            self.new_shape = None

    def translate_state(self, state):
        raise NotImplementedError()
    

    def _get_shape(self, original_shape=None):
        raise NotImplementedError()
    

    def get_shape(self, original_shape=None):
        
        if original_shape is None and self.new_shape is not None:
        
            return self.new_shape
        
        else:
            return self._get_shape(original_shape)

    


class TranslatorSequence(Translator):

    '''A strategy that translates data with a preditermined type and shape to another'''

    parameters_signature = {
        "translators_sequence" : ComponentListInputSignature()
    }

    def _proccess_input_internal(self):

        super()._proccess_input_internal()

        self.translators_sequence : list[Translator] = self.get_input_value("translators_sequence")

        self._setup_translators_in_sequence_input()

        self._setup_sequence_state_cache()

        for translator in self.translators_sequence:
            translator.proccess_input_if_not_proccesd()

    
    def _setup_translators_in_sequence_input(self):
        

        if self.was_custom_value_passed_for_input("in_place_translation"):

            for translator in self.translators_sequence:
                translator.pass_input_if_no_value("in_place_translation", self.in_place_translation)


        if self.was_custom_value_passed_for_input("buffered_operations"):

            for translator in self.translators_sequence:
                translator.pass_input_if_no_value("buffered_operations", self.buffered_operations)



    def _setup_shape_cache(self): # we overwrite this method to do nothing as we want to have control over how the cache is proccessed
        pass

    
    def _setup_sequence_state_cache(self):

        if self.original_shape is not None:

            current_shape = self.original_shape

            for translator in self.translators_sequence:

                translator.pass_input({"original_shape" : current_shape})
                translator.proccess_input_if_not_proccesd()
                current_state = translator.get_shape()

            self.new_state = current_shape

        else:
            self.new_state = None



    def translate_state(self, state):
        
        current_state = state

        for translator in self.translators_sequence:
            current_state = translator.translate_state(current_state)

        return current_state
    
    
    def _get_shape(self, original_shape):

        current_shape = original_shape

        for translator in self.translators_sequence:
            current_shape = translator.get_shape(current_shape)

        return current_shape