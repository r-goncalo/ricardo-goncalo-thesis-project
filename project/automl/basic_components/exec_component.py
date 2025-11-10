from automl.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automl.basic_components.state_management import save_state
from automl.utils.smart_enum import SmartEnum
from automl.loggers.logger_component import flush_text_of_all_loggers_and_children
from ..component import InputSignature, Component, requires_input_proccess

from abc import abstractmethod
from enum import Enum

from typing import final

class State(SmartEnum): #an enumerator to track state of executable component
        IDLE = 0
        RUNNING = 1
        OVER = 2
        ERROR = 4

# EXECUTABLE COMPONENT --------------------------

class ExecComponent(Component):
    
    parameters_signature = {
        
            "times_to_run" : InputSignature(mandatory=False, description="The number of times to run the component"),
            "save_state_on_run_end" : InputSignature(default_value=True, ignore_at_serialization=True),
            "save_dataframes_on_run_end" : InputSignature(default_value=True, ignore_at_serialization=True)
    
    }

    exposed_values = {"running_state" : State.IDLE, "times_ran" : 0}    
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        if "times_to_run" not in self.input:
            self._times_to_run = None
        else:
            self._times_to_run = self.get_input_value("times_to_run")
        
        self.__save_state_on_run_end = self.get_input_value("save_state_on_run_end")
        self.__save_dataframes_on_run_end = self.get_input_value("save_dataframes_on_run_end")

        


    # METHODS TO OVERRIDE --------------------------------


    def _algorithm(self): #the algorithm of the component
        pass
    
    def _pre_algorithm(self): # a component may extend this for certain behaviours
        self.values["running_state"] = State.RUNNING
        
    def _pos_algorithm(self): # a component may extend this for certain behaviours
        
        self.values["times_ran"] += 1
        
        if self._times_to_run != None and self.values["times_ran"] < self._times_to_run:
            self.values["running_state"] = State.IDLE
            
        else:
            self.values["running_state"] = State.OVER
        
        if self.__save_state_on_run_end:
            save_state(self)
        
        if self.__save_dataframes_on_run_end:
            save_all_dataframes_of_component_and_children(self)
            flush_text_of_all_loggers_and_children(self)
            
        

    def __on_exception(self, exception):
        '''To be called when a non treated exception happens'''
        self._deal_with_exception(exception)
    
    def _deal_with_exception(self, exception):
        '''Called internally when an exception happens'''
        pass

    # RUNNABLE METHOD --------------------------------
    
    @requires_input_proccess
    @final
    def run_all(self):
        
        '''Runs the algorithm the number of times specified in the input.'''
        
        if self._times_to_run == None:
            return self.run()
        
        else:
            for i in range(self._times_to_run - self.values["times_ran"]):
                self.run()
                
            return self.get_output()
    
    @requires_input_proccess
    @final
    def run(self): #the universal execution flow for all components
        
        '''Runs the specified algorithm once.'''
         
        try:
            self._pre_algorithm()
            self._algorithm()
            self._pos_algorithm()
            return self.get_output()
        
        except Exception as e:
            self.values["running_state"] = State.ERROR
            self.__on_exception(e)
            raise e # TODO: decide on this, are the elements that run this responsible for caughting the exception?
    
